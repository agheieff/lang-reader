"""Language packs: rules, grammar catalogs and word lists found by directory.

A pack directory ``<pack>/`` may hold ``rules/*.toml``, ``grammar/*.toml`` and
``wordlist/wordlist.tsv`` (with ``wordlist/pack.toml`` naming its ``source``). Packs are read from
the repository's ``packs/`` directory and any directories named by ``ARC_LANG_PACK_DIRS``;
adding a language is adding a directory. Each pack carries its own LICENSE and SOURCES.md.
"""

from __future__ import annotations

from pathlib import Path

from server import settings

BUNDLED_PACKS = settings.PROJECT_ROOT / "packs"


def pack_roots() -> tuple[Path, ...]:
    return (BUNDLED_PACKS, *settings.load().pack_dirs)


def pack_files(kind: str) -> tuple[Path, ...]:
    """Every ``<root>/<pack>/<kind>/*.toml``, in a stable order."""

    return tuple(
        sorted(
            path
            for root in pack_roots()
            for path in root.glob(f"*/{kind}/*.toml")
            if path.is_file()
        )
    )


def wordlist_directories() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for root in pack_roots()
            for path in root.glob("*/wordlist")
            if (path / "wordlist.tsv").is_file()
        )
    )
