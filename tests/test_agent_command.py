from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from server.agent_command import read_usage
from server.generation_callback import CallbackInvocation
from server.model_callback import ProviderCallback
from server.providers import CommandProvider, ModelConfig


def _agent(
    command: list[str],
    *,
    timeout: int = 30,
    passthrough: tuple[str, ...] = (),
) -> ProviderCallback:
    provider = CommandProvider(
        kind="command",
        name="test",
        command=command,
        timeout_seconds=timeout,
        pass_env=list(passthrough),
    )
    config = ModelConfig(providers={"test": provider}, stages={"default": ("test",)})
    return ProviderCallback(config, unavailable_exit_code=3)


def _script(tmp_path: Path, body: str) -> list[str]:
    path = tmp_path / "agent.py"
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


def test_command_gets_canonical_request_on_stdin_and_protocol_environment(
    invocation: CallbackInvocation, tmp_path: Path
) -> None:
    command = _script(
        tmp_path,
        "import json, os, sys\n"
        "request = json.loads(sys.stdin.read())\n"
        "env = {k: v for k, v in os.environ.items() if k.startswith('ARC_LANG_')}\n"
        "print(json.dumps({'stage': request['stage'], 'task_id': request['task_id'], 'env': env,\n"
        "                  'prompt': open(env['ARC_LANG_CALLBACK_PROMPT_PATH']).read()}))\n",
    )
    result = _agent(command).run(invocation)

    assert result.exit_code == 0 and not result.deferred
    seen = json.loads(result.payload)
    env = seen["env"]
    assert (seen["stage"], seen["task_id"]) == ("prose", 1)
    assert env["ARC_LANG_PROTOCOL"] == "1" and env["ARC_LANG_CALLBACK_PROTOCOL"] == "4"
    assert env["ARC_LANG_CALLBACK_STAGE"] == "prose"
    assert env["ARC_LANG_JOB_ID"] == "es-es-en:1"
    assert len(env["ARC_LANG_REQUEST_SHA256"]) == 64
    assert env["ARC_LANG_CALLBACK_RESPONSE_SCHEMA_PATH"].endswith("prose-response-schema.json")
    assert seen["prompt"].startswith("Write the frozen lesson prose for a language-learning")


def test_response_file_wins_over_stdout_and_usage_is_recorded(
    invocation: CallbackInvocation, tmp_path: Path
) -> None:
    command = _script(
        tmp_path,
        "import json, os\n"
        "open(os.environ['ARC_LANG_CALLBACK_RESPONSE_PATH'], 'w').write('{\"from\": \"file\"}')\n"
        "json.dump({'input_tokens': 120, 'cached_input_tokens': 100, 'output_tokens': 7,\n"
        "           'ignored': 9, 'reasoning_output_tokens': -1},\n"
        "          open(os.environ['ARC_LANG_CALLBACK_USAGE_PATH'], 'w'))\n"
        'print(\'{"from": "stdout"}\')\n',
    )
    result = _agent(command).run(invocation)

    assert json.loads(result.payload) == {"from": "file"}
    assert result.usage == {"input_tokens": 120, "cached_input_tokens": 100, "output_tokens": 7}


def test_stale_usage_from_an_earlier_call_is_never_reported(
    invocation: CallbackInvocation, tmp_path: Path
) -> None:
    assert invocation.usage_path is not None
    invocation.usage_path.write_text('{"input_tokens": 999}', encoding="utf-8")
    result = _agent(_script(tmp_path, "print('{}')")).run(invocation)
    assert result.usage == {}


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '"x"', '{"input_tokens": true}'])
def test_malformed_usage_is_empty(tmp_path: Path, text: str) -> None:
    path = tmp_path / "usage.json"
    path.write_text(text, encoding="utf-8")
    assert read_usage(path) == {}
    assert read_usage(None) == {}
    assert read_usage(tmp_path / "missing.json") == {}


@pytest.mark.parametrize("exit_code,deferred", [(0, False), (1, False), (3, True), (124, False)])
def test_only_the_unavailable_exit_defers(
    invocation: CallbackInvocation, tmp_path: Path, exit_code: int, deferred: bool
) -> None:
    command = _script(tmp_path, f"import sys; sys.exit({exit_code})")
    result = _agent(command).run(invocation)
    assert (result.exit_code, result.deferred) == (exit_code, deferred)


def test_a_command_that_outlives_the_timeout_is_stopped(
    invocation: CallbackInvocation, tmp_path: Path
) -> None:
    command = _script(tmp_path, "import time; time.sleep(60)")
    result = _agent(command, timeout=1).run(invocation)
    assert result.exit_code == 1 and "timed out" in result.log and not result.deferred


def test_a_missing_executable_is_a_failed_call_not_a_crash(
    invocation: CallbackInvocation, tmp_path: Path
) -> None:
    result = _agent([str(tmp_path / "no-such-agent")]).run(invocation)
    assert result.exit_code == 1 and result.payload == ""


def test_child_environment_is_an_allowlist(
    invocation: CallbackInvocation, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "secret-that-must-not-leak")
    monkeypatch.setenv("MY_ADAPTER_HOME", "/adapter")
    command = _script(tmp_path, "import json, os; print(json.dumps(dict(os.environ)))")
    plain = json.loads(_agent(command).run(invocation).payload)
    assert "EXAMPLE_API_KEY" not in plain and "MY_ADAPTER_HOME" not in plain
    shared = _agent(command, passthrough=("MY_ADAPTER_HOME",))
    assert json.loads(shared.run(invocation).payload)["MY_ADAPTER_HOME"] == "/adapter"
