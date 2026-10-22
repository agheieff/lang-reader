"""Pure Han-script identity helpers shared by tracking and learning projections."""

from __future__ import annotations

import unicodedata

from server.language_packs import language_pack

_HAN_RANGES = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x2B820, 0x2CEAF),  # Extension E
    (0x2CEB0, 0x2EBEF),  # Extension F
    (0x2EBF0, 0x2EE5F),  # Extension I
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    (0x30000, 0x3134F),  # Extension G
    (0x31350, 0x323AF),  # Extension H
)


def character_tracking_available(language_tag: str) -> bool:
    """Return whether the profile's declarative language pack represents Chinese."""

    return language_pack(language_tag).id == "zh"


def is_han_character(character: str) -> bool:
    """Recognize one Han code point without conflating simplified and traditional forms."""

    if len(character) != 1:
        return False
    codepoint = ord(character)
    if codepoint == 0x3007:  # Ideographic number zero
        return True
    return any(start <= codepoint <= end for start, end in _HAN_RANGES)


def han_characters(text: str) -> tuple[str, ...]:
    """Return literal normalized Han code points in display order."""

    return tuple(
        character for character in unicodedata.normalize("NFC", text) if is_han_character(character)
    )


def distinct_han_characters(text: str) -> tuple[str, ...]:
    """Return literal Han identities once each, preserving first occurrence order."""

    return tuple(dict.fromkeys(han_characters(text)))
