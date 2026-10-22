"""Compact, honest language-variant markers for the profile picker."""

from __future__ import annotations

import re
from dataclasses import dataclass

from server.workspaces import Workspace


@dataclass(frozen=True)
class LanguageMarker:
    symbol: str
    description: str


@dataclass(frozen=True)
class ProfileOption:
    profile_id: str
    marker: LanguageMarker
    visible_label: str
    accessible_label: str


_REGION_NAMES = {
    "AT": "Austria",
    "BR": "Brazil",
    "CA": "Canada",
    "CH": "Switzerland",
    "CN": "China",
    "DE": "Germany",
    "ES": "Spain",
    "FR": "France",
    "GB": "United Kingdom",
    "HK": "Hong Kong",
    "MO": "Macao",
    "MX": "Mexico",
    "PT": "Portugal",
    "TW": "Taiwan",
    "US": "United States",
}
_NUMERIC_REGIONS = {
    "419": ("🌎", "Latin America and the Caribbean"),
}
_SCRIPTS = {
    "Arab": ("ع", "Arabic script"),
    "Cyrl": ("А", "Cyrillic script"),
    "Deva": ("अ", "Devanagari script"),
    "Hans": ("简", "Simplified Han script"),
    "Hant": ("繁", "Traditional Han script"),
    "Jpan": ("日", "Japanese script"),
    "Kore": ("한", "Korean script"),
    "Latn": ("Aa", "Latin script"),
}
_LANGUAGE_NAMES = {
    "de-de": "German",
    "es-419": "Spanish (Latin America)",
    "es-es": "Spanish (Spain)",
    "zh-hans": "Chinese (Simplified)",
    "zh-hant": "Chinese (Traditional)",
}
_BASE_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}
_SCRIPT_NAMES = {
    "Hans": "Simplified",
    "Hant": "Traditional",
}
_LANGUAGE_CODE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$")


def _country_flag(region: str) -> str:
    return "".join(chr(0x1F1E6 + ord(character) - ord("A")) for character in region)


def language_marker(language_tag: str) -> LanguageMarker:
    base_language = language_tag.split("-", 1)[0].lower()
    subtags = language_tag.split("-")[1:]
    script = next((part.title() for part in subtags if len(part) == 4 and part.isalpha()), None)
    region = next(
        (
            part.upper()
            for part in subtags
            if (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit())
        ),
        None,
    )
    symbols: list[str] = []
    descriptions: list[str] = []

    if base_language == "zh" and script == "Hans" and region is None:
        return LanguageMarker("🇨🇳", "Simplified Chinese (China used as the picker marker)")

    if region and len(region) == 2:
        symbols.append(_country_flag(region))
        descriptions.append(_REGION_NAMES.get(region, region))
    elif region:
        symbol, description = _NUMERIC_REGIONS.get(region, ("🌐", f"Region {region}"))
        symbols.append(symbol)
        descriptions.append(description)

    if script:
        symbol, description = _SCRIPTS.get(script, (script, f"{script} script"))
        symbols.append(symbol)
        descriptions.append(description)

    if not symbols:
        return LanguageMarker("🌐", "No region or script specified")
    return LanguageMarker(" ".join(symbols), ", ".join(descriptions))


def language_name(language_tag: str, fallback: str) -> str:
    """Return a human name without exposing a raw BCP-47 tag in the visible picker."""

    exact = _LANGUAGE_NAMES.get(language_tag.casefold())
    if exact is not None:
        return exact

    parts = language_tag.split("-")
    base = _BASE_LANGUAGE_NAMES.get(parts[0].casefold())
    if base is not None:
        script = next(
            (part.title() for part in parts[1:] if len(part) == 4 and part.isalpha()), None
        )
        region = next(
            (
                part.upper()
                for part in parts[1:]
                if (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit())
            ),
            None,
        )
        qualifiers: list[str] = []
        if script is not None:
            qualifiers.append(_SCRIPT_NAMES.get(script, f"{script} script"))
        if region is not None:
            if region in _NUMERIC_REGIONS:
                qualifiers.append(_NUMERIC_REGIONS[region][1])
            else:
                qualifiers.append(_REGION_NAMES.get(region, "Regional variant"))
        return f"{base} ({', '.join(qualifiers)})" if qualifiers else base

    cleaned_fallback = fallback.strip()
    if cleaned_fallback and _LANGUAGE_CODE.fullmatch(cleaned_fallback) is None:
        return cleaned_fallback
    return "Language"


def profile_option(workspace: Workspace) -> ProfileOption:
    marker = language_marker(workspace.learning_language)
    visible_label = language_name(workspace.learning_language, workspace.label)
    return ProfileOption(
        profile_id=workspace.profile_id,
        marker=marker,
        visible_label=visible_label,
        accessible_label=f"{visible_label}; {marker.description}",
    )
