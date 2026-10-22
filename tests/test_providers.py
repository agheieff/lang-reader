from __future__ import annotations

import re
from pathlib import Path

import pytest

from server import settings
from server.providers import (
    AnthropicProvider,
    CommandProvider,
    OpenAIProvider,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "lang-reader.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _toml_fences(path: Path) -> list[str]:
    return re.findall(r"```toml\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)


def _config_dir(tmp_path: Path, text: str) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for key_file in re.findall(r'api_key_file = "([^"]+)"', text):
        (config_dir / key_file).write_text("k\n", encoding="utf-8")
    (config_dir / "lang-reader.toml").write_text(text, encoding="utf-8")
    return config_dir / "lang-reader.toml"


def test_the_committed_example_parses(tmp_path: Path) -> None:
    text = (ROOT / "lang-reader.example.toml").read_text(encoding="utf-8")
    config = load_config(_config_dir(tmp_path, text), {})
    assert [provider.name for provider in config.chain("prose")] == ["main"]


def test_every_provider_block_in_the_readme_is_a_working_configuration(tmp_path: Path) -> None:
    fences = [text for text in _toml_fences(ROOT / "README.md") if "[providers." in text]
    assert len(fences) >= 7  # OpenAI, Anthropic, OpenRouter, DeepSeek, local, command
    for index, text in enumerate(fences):
        config = load_config(_config_dir(tmp_path / str(index), text), {})
        for stage in settings.CALLBACK_STAGES:
            assert config.chain(stage)


def test_uncommenting_the_spare_providers_in_the_example_parses(tmp_path: Path) -> None:
    lines = (ROOT / "lang-reader.example.toml").read_text(encoding="utf-8").splitlines()
    start = lines.index("# [providers.claude]")
    end = next(i for i, line in enumerate(lines) if line.startswith("# Stages:"))
    spare = [line[2:] if line.startswith("# ") else "" for line in lines[start:end]]
    assigned = [line[2:] for line in lines if line.startswith("# lexical = ")]
    text = "\n".join([*lines[:start], *spare, "[stages]", 'default = "main"', *assigned])
    config = load_config(_config_dir(tmp_path, text), {})
    assert {"main", "claude", "cheap", "local", "custom"} <= set(config.providers)
    assert [provider.name for provider in config.chain("lexical")] == ["cheap", "main"]


def test_all_three_kinds_parse_with_defaults_and_resolved_keys(tmp_path: Path) -> None:
    (tmp_path / "key.txt").write_text("  file-key \n", encoding="utf-8")
    path = _write(
        tmp_path,
        """
[providers.a]
kind = "openai"
base_url = "https://api.example.com/v1/"
model = "m"
api_key_file = "key.txt"

[providers.b]
kind = "anthropic"
model = "claude"
api_key_env = "B_KEY"

[providers.c]
kind = "command"
command = ["my-call", "--flag"]
pass_env = ["X"]

[stages]
default = "a"
lexical = ["c", "b"]
""",
    )
    config = load_config(path, {"B_KEY": "env-key"})
    a, b, c = config.providers["a"], config.providers["b"], config.providers["c"]
    assert isinstance(a, OpenAIProvider) and isinstance(b, AnthropicProvider)
    assert isinstance(c, CommandProvider)
    assert a.base_url == "https://api.example.com/v1" and a.api_key == "file-key"
    assert a.schema_mode == "schema" and a.timeout_seconds == 600
    assert b.base_url == "https://api.anthropic.com" and b.api_key == "env-key"
    assert b.max_output_tokens == 16000
    assert c.command == ["my-call", "--flag"] and c.pass_env == ["X"]
    assert [p.name for p in config.chain("prose")] == ["a"]
    assert [p.name for p in config.chain("lexical")] == ["c", "b"]
    assert "file-key" not in repr(a) and "file-key" not in a.model_dump_json()


def test_a_single_provider_needs_no_stages_table(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        '[providers.only]\nkind = "openai"\nbase_url = "http://localhost:8080/v1"\nmodel = "m"\n',
    )
    config = load_config(path, {})
    assert config.providers["only"].kind == "openai"  # a local server needs no key
    assert [p.name for p in config.chain("grammar")] == ["only"]


@pytest.mark.parametrize(
    "text,message",
    [
        ("", "at least one"),
        ("not toml [", "not valid TOML"),
        ("surprise = 1\n[providers.a]\nkind='command'\ncommand=['x']", "unknown top-level"),
        ("[providers.a]\nkind='ftp'\n", "providers.a"),
        ("[providers.a]\nkind='command'\ncommand=[]\n", "command"),
        ("[providers.a]\nkind='command'\ncommand=['x']\nextra=1\n", "extra"),
        ('[providers."a b"]\nkind="command"\ncommand=["x"]\n', "may use letters"),
        (
            "[providers.a]\nkind='openai'\nbase_url='ftp://h'\nmodel='m'\n",
            r"http\(s\) URL",
        ),
        (
            "[providers.a]\nkind='openai'\nbase_url='http://h/v1'\nmodel='m'\napi_key='sk-x'\n",
            "inline api_key",
        ),
        (
            "[providers.a]\nkind='openai'\nbase_url='http://h/v1'\nmodel='m'\nschema_mode='xml'\n",
            "schema_mode",
        ),
        (
            "[providers.a]\nkind='openai'\nbase_url='http://h/v1'\nmodel='m'\n"
            "api_key_file='nope.key'\n",
            "cannot read api_key_file",
        ),
        (
            "[providers.a]\nkind='openai'\nbase_url='http://h/v1'\nmodel='m'\n"
            "api_key_env='UNSET_VAR'\n",
            "is empty",
        ),
        (
            "[providers.a]\nkind='openai'\nbase_url='http://h/v1'\nmodel='m'\n"
            "api_key_env='K'\napi_key_file='f'\n",
            "api_key_file or api_key_env",
        ),
        ("[providers.a]\nkind='anthropic'\nmodel='m'\n", "anthropic needs a key"),
        (
            "[providers.a]\nkind='command'\ncommand=['x']\ntimeout_seconds=0\n",
            "timeout_seconds",
        ),
        (
            "[providers.a]\nkind='command'\ncommand=['x']\n[providers.b]\nkind='command'\n"
            "command=['y']\n",
            "set stages.default",
        ),
        (
            "[providers.a]\nkind='command'\ncommand=['x']\n[stages]\ndefault='ghost'\n",
            "unknown providers",
        ),
        (
            "[providers.a]\nkind='command'\ncommand=['x']\n[stages]\ndefault='a'\nlexicon='a'\n",
            "unknown stage",
        ),
        (
            "[providers.a]\nkind='command'\ncommand=['x']\n[stages]\ndefault=['a','a']\n",
            "twice",
        ),
    ],
)
def test_invalid_configuration_fails_with_a_reason(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, text), {})


def test_a_missing_file_points_at_the_example(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"lang-reader\.example\.toml"):
        load_config(tmp_path / "lang-reader.toml", {})


def test_settings_locate_the_file_from_the_environment() -> None:
    assert settings.load({"ARC_LANG_CONFIG": "/etc/lr.toml"}).agent.config_path == Path(
        "/etc/lr.toml"
    )
    assert settings.load({}).agent.config_path == settings.PROJECT_ROOT / "lang-reader.toml"
