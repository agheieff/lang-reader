"""The documentation must not drift from the code it describes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("README.md", "AGENTS.md", "packs/README.md", "extras/tts-qwen/README.md")
_NAME = re.compile(r"ARC_LANG_[A-Z0-9_]+")


def _names(text: str) -> set[str]:
    return set(_NAME.findall(text))


def test_every_setting_the_code_reads_is_documented() -> None:
    settings_source = (ROOT / "server" / "settings.py").read_text(encoding="utf-8")
    read = {name for name in _names(settings_source) if f'"{name}"' in settings_source}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tts_readme = (ROOT / "extras" / "tts-qwen" / "README.md").read_text(encoding="utf-8")
    missing = {name for name in read if name not in readme and name not in tts_readme}
    assert not missing, f"not documented: {sorted(missing)}"


def test_env_example_names_only_real_settings() -> None:
    known = _names((ROOT / "server" / "settings.py").read_text(encoding="utf-8"))
    example = _names((ROOT / ".env.example").read_text(encoding="utf-8"))
    assert example and example <= known, sorted(example - known)


def test_the_readme_does_not_mention_removed_settings() -> None:
    for document in DOCUMENTS:
        text = (ROOT / document).read_text(encoding="utf-8")
        assert not re.search(r"ARC_LANG_(HTTP_|AGENT_COMMAND|AGENT_ENV)", text), document


def _anchors(markdown: str) -> set[str]:
    anchors = set()
    for heading in re.findall(r"^#{1,6} (.+)$", markdown, re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.lower().replace("`", "")).replace(" ", "-")
        anchors.add(slug)
    return anchors


@pytest.mark.parametrize("document", DOCUMENTS)
def test_relative_links_and_anchors_resolve(document: str) -> None:
    path = ROOT / document
    text = path.read_text(encoding="utf-8")
    unfenced = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    for target in re.findall(r"\]\(([^)\s]+)\)", unfenced):
        if re.match(r"[a-z]+:", target):
            continue  # external link
        file_part, _, anchor = target.partition("#")
        linked = (path.parent / file_part).resolve() if file_part else path
        assert linked.exists(), f"{document}: {target} does not exist"
        if anchor and linked.suffix == ".md":
            assert anchor in _anchors(linked.read_text(encoding="utf-8")), f"{document}: {target}"
