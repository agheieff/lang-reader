"""Shared vocabulary: the strict base model, language-tag handling and the literal aliases."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    StringConstraints,
)

LESSON_SCHEMA_VERSION: Literal[1] = 1
CefrLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
LevelSource = Literal["unknown", "self_reported", "estimated"]
CalibrationStatus = Literal["unstarted", "collecting", "rough", "stable"]
TermKnowledgeBand = Literal["familiar", "expected", "uncertain", "focus", "incidental"]
FeedbackTag = Literal[
    "shorter",
    "longer",
    "easier",
    "more_challenging",
    "more_grammar",
    "less_grammar",
    "same_topic",
    "new_topic",
]
# Broad subject areas a generated text declares, so repeats can be detected without a model.
SubjectDomain = Literal[
    "home-household",
    "food-cooking",
    "travel-transport",
    "work-professions",
    "money-shopping",
    "health-body",
    "sport-fitness",
    "music",
    "visual-art-design",
    "literature-language",
    "history",
    "nature-animals-plants",
    "weather-seasons",
    "science-space",
    "technology-devices",
    "city-places",
    "school-education",
    "family-friends",
    "festivals-traditions",
    "crafts-repair",
    "games-hobbies",
    "media-news",
    "society-law",
    "mathematics-puzzles",
]
GenerationTaskState = Literal["pending", "running", "completed", "failed"]
GenerationRequestKind = Literal["queue_fill", "topic_request"]
TextStatus = Literal["queued", "in_progress", "read", "skipped"]
EventType = Literal[
    "lesson.started",
    "term.revealed",
    "translation.revealed",
    "lesson.completed",
    "lesson.rated",
]

_PROPER_NOUN_POS = frozenset({"proper noun", "propernoun", "propn", "name"})


def is_proper_noun_pos(value: str) -> bool:
    normalized = "".join(character if character.isalnum() else " " for character in value)
    return " ".join(normalized.casefold().split()) in _PROPER_NOUN_POS


def _normalize_language_tag(value: str) -> str:
    parts = value.strip().split("-")
    normalized = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            normalized.append(part.title())
        elif (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            normalized.append(part.upper())
        else:
            normalized.append(part.lower())
    return "-".join(normalized)


LanguageTag = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$"),
    AfterValidator(_normalize_language_tag),
]
LANGUAGE_TAG_DESCRIPTION = (
    "BCP-47-like language tag; examples: de-DE, es-ES, es-419, zh-Hans, zh-Hant."
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
