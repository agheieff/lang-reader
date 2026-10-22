"""Frequency-ordered vocabulary lists used to propose new words.

A list is the ``wordlist/`` directory of a language pack (see ``packs.py``): ``wordlist.tsv``
with columns lemma, pronunciation, gloss, frequency_rank, level, and ``pack.toml`` naming the
``source`` and the base ``languages`` it serves. Lists carry their own licence; the Chinese one
is a separate CC BY-SA 4.0 data pack that can be left out or fetched by
``scripts/build_zh_word_list.py``.
"""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[import-not-found]

from server.packs import wordlist_directories


@dataclass(frozen=True)
class WordListEntry:
    lemma: str
    pronunciation: str | None
    gloss: str
    frequency_rank: int | None
    level: int | None
    source: str


def normalized_lemma(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def word_list(language_tag: str) -> tuple[WordListEntry, ...]:
    """The list for a language tag's base language; BCP-47 variants share their base's list."""

    base = language_tag.strip().replace("_", "-").split("-")[0].casefold()
    return _load(base, wordlist_directories())


def word_list_summaries() -> list[dict[str, object]]:
    """Every discovered list, fully parsed (a malformed one raises), for ``lang pack check``."""

    summaries: list[dict[str, object]] = []
    for directory in wordlist_directories():
        with (directory / "pack.toml").open("rb") as stream:
            description = tomllib.load(stream)
        entries = _read(directory / "wordlist.tsv", str(description["source"]))
        summaries.append(
            {
                "directory": str(directory),
                "languages": list(description["languages"]),
                "source": description["source"],
                "entries": len(entries),
            }
        )
    return summaries


@lru_cache(maxsize=8)
def _load(base: str, directories: tuple[Path, ...]) -> tuple[WordListEntry, ...]:
    for directory in directories:
        with (directory / "pack.toml").open("rb") as stream:
            description = tomllib.load(stream)
        if base in description["languages"]:
            return _read(directory / "wordlist.tsv", str(description["source"]))
    return ()


def _read(path: Path, source: str) -> tuple[WordListEntry, ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        return tuple(
            WordListEntry(
                lemma=unicodedata.normalize("NFC", row["lemma"]),
                pronunciation=row["pronunciation"] or None,
                gloss=row["gloss"],
                frequency_rank=int(row["frequency_rank"]) if row["frequency_rank"] else None,
                level=int(row["level"]) if row["level"] else None,
                source=source,
            )
            for row in rows
        )
