from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "packs"


def _check(*tags: str, packs: Path | None = EXAMPLE) -> dict[str, object]:
    env = {**os.environ, "ARC_LANG_PACK_DIRS": str(packs) if packs else ""}
    result = subprocess.run(
        [sys.executable, "-m", "server.cli", "pack", "check", *tags],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report: dict[str, object] = json.loads(result.stdout)
    return report


def test_the_example_pack_is_found_through_pack_dirs_and_serves_variants() -> None:
    report = _check("fr-CA", "de-AT", "pt-BR")
    languages = report["languages"]
    assert isinstance(languages, dict)
    assert languages["fr-CA"] == {
        "rules": "fr",
        "generation_guidance": 2,
        "grammar_catalog": "fr",
        "word_list_entries": 2,
    }
    assert languages["de-AT"]["rules"] == "de"  # a variant uses its base language's pack
    assert languages["pt-BR"]["rules"] == "generic"  # no pack: safe defaults, no catalog
    assert languages["pt-BR"]["grammar_catalog"] is None


def test_without_the_extra_directory_the_language_has_no_pack() -> None:
    report = _check("fr-CA", packs=None)
    languages = report["languages"]
    assert isinstance(languages, dict)
    assert languages["fr-CA"]["rules"] == "generic"


def test_a_malformed_pack_fails_the_check(tmp_path: Path) -> None:
    broken = tmp_path / "xx" / "rules"
    broken.mkdir(parents=True)
    (broken / "xx.toml").write_text('schema_version = 1\nid = "xx"\n', encoding="utf-8")
    env = {**os.environ, "ARC_LANG_PACK_DIRS": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-m", "server.cli", "pack", "check"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1 and "error:" in result.stderr


def test_the_readme_shows_the_example_files_verbatim() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative in ("rules/fr.toml", "grammar/fr.toml", "wordlist/pack.toml"):
        text = (EXAMPLE / "fr" / relative).read_text(encoding="utf-8")
        assert text in readme, relative
