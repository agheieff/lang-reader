"""Build packs/zh/wordlist/wordlist.tsv from drkameleon/complete-hsk-vocabulary (MIT).

Usage: uv run python scripts/build_zh_word_list.py path/to/complete.min.json
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "packs" / "zh" / "wordlist" / "wordlist.tsv"


def _level(levels: list[str]) -> int | None:
    """HSK 3.0 level (7 covers 7-9), falling back to HSK 2.0."""

    new = [int(item[1:]) for item in levels if item.startswith("n")]
    old = [int(item[1:]) for item in levels if item.startswith("o")]
    return min(new) if new else (min(old) if old else None)


_NOISE = re.compile(
    r"^(surname |(old |archaic )?variant of |see |used in |abbr\. |\(old\) |"
    r"[A-Z][a-z]+ (District|County|Province|City)\b)"
)


def _useful_meanings(values: list[str]) -> list[str]:
    """Prefer ordinary senses over surnames, variant pointers, and place names."""

    meanings = [value.strip() for value in values if value.strip()]
    useful = [value for value in meanings if not _NOISE.search(value)]
    return useful or meanings


def _pinyin(value: str) -> str:
    # Lower-case tone-mark pinyin with one space between syllables, as the zh pack expects.
    return " ".join(unicodedata.normalize("NFC", value).lower().split())


def main(source: Path) -> None:
    entries = json.loads(source.read_text(encoding="utf-8"))
    rows: dict[str, tuple[str, str, str, int, int]] = {}
    for entry in entries:
        lemma = unicodedata.normalize("NFC", entry["s"]).strip()
        level = _level(entry.get("l", []))
        forms = entry.get("f") or []
        if not lemma or level is None or not forms:
            continue
        # Forms differ by traditional spelling or reading; prefer the one with ordinary senses.
        form = max(
            forms,
            key=lambda item: sum(
                1 for value in item.get("m", []) if value.strip() and not _NOISE.search(value)
            ),
        )
        pronunciation = _pinyin(form.get("i", {}).get("y", ""))
        meanings = _useful_meanings(form.get("m", []))
        gloss = "; ".join(meanings[:3]).replace("\t", " ").replace("\n", " ")
        rank = int(entry.get("q") or 0) or 999_999
        row = (lemma, pronunciation, gloss, rank, level)
        previous = rows.get(lemma)
        if previous is None or (level, rank) < (previous[4], previous[3]):
            rows[lemma] = row
    ordered = sorted(rows.values(), key=lambda row: (row[3], row[4], row[0]))
    lines = ["lemma\tpronunciation\tgloss\tfrequency_rank\tlevel"]
    lines += [f"{a}\t{b}\t{c}\t{d}\t{e}" for a, b, c, d, e in ordered]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(ordered)} entries to {OUTPUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
