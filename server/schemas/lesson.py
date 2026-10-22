"""The stored lesson document and its calibration records."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from server.grammar_catalog import grammar_catalog, grammar_construction
from server.schemas.base import (
    LANGUAGE_TAG_DESCRIPTION,
    LESSON_SCHEMA_VERSION,
    CefrLevel,
    LanguageTag,
    StrictModel,
)


class LessonTerm(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    lemma: str = Field(min_length=1, max_length=200)
    pos: str = Field(min_length=1, max_length=40)
    gloss: str = Field(min_length=1, max_length=500)
    pronunciation: str | None = Field(default=None, max_length=300)
    frequency_rank: int | None = Field(default=None, ge=1)

    @field_validator("key", "lemma", "pos", "gloss", "pronunciation")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LessonRun(StrictModel):
    text: str = Field(min_length=1)
    term: LessonTerm | None = None
    pronunciation: str | None = Field(default=None, max_length=300)

    @field_validator("pronunciation")
    @classmethod
    def strip_pronunciation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_term_surface(self) -> LessonRun:
        if self.term is not None and not self.text.strip():
            raise ValueError("a term run must contain visible text")
        return self


class GrammarOccurrence(StrictModel):
    """One contextual use of a cataloged construction, anchored to whole lesson runs."""

    key: str = Field(
        min_length=1,
        max_length=200,
        description="Stable lesson-unique identity for this occurrence.",
    )
    construction_key: str = Field(
        min_length=1,
        max_length=100,
        description="Stable key from the tracked grammar catalog.",
    )
    run_start: int = Field(
        ge=0,
        description="Inclusive zero-based index of the first covered sentence run.",
    )
    run_end: int = Field(
        ge=1,
        description="Exclusive zero-based index after the last covered sentence run.",
    )
    note: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Optional contextual explanation of this use; omit when the catalog description "
            "already suffices."
        ),
    )

    @field_validator("key", "construction_key")
    @classmethod
    def strip_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("note")
    @classmethod
    def strip_optional_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("construction_key")
    @classmethod
    def require_catalog_membership(cls, value: str) -> str:
        try:
            grammar_construction(value)
        except LookupError as error:
            raise ValueError(str(error)) from error
        return value

    @model_validator(mode="after")
    def validate_run_range(self) -> GrammarOccurrence:
        if self.run_end <= self.run_start:
            raise ValueError("grammar occurrence run_end must be greater than run_start")
        return self


class LessonSentence(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    runs: list[LessonRun] = Field(min_length=1)
    translation: str | None = None
    grammar: list[GrammarOccurrence] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Optional grammar occurrences anchored to half-open ranges of this sentence's runs."
        ),
    )

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_grammar_ranges(self) -> LessonSentence:
        occurrence_keys = [occurrence.key for occurrence in self.grammar]
        if len(occurrence_keys) != len(set(occurrence_keys)):
            raise ValueError("grammar occurrence keys must be unique within a sentence")
        for occurrence in self.grammar:
            if occurrence.run_end > len(self.runs):
                raise ValueError(
                    f"grammar occurrence range exceeds sentence runs: {occurrence.key}"
                )
        return self

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)


class LessonBlock(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    sentences: list[LessonSentence] = Field(min_length=1)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


def _collect_sentence_terms(
    sentence: LessonSentence, terms: dict[str, LessonTerm], *, generated: bool = False
) -> None:
    for run in sentence.runs:
        if run.term is None:
            continue
        previous = terms.get(run.term.key)
        if previous is not None and previous != run.term:
            qualifier = " generated" if generated else ""
            raise ValueError(f"term key has conflicting{qualifier} definitions: {run.term.key}")
        terms[run.term.key] = run.term


class CalibrationProbe(StrictModel):
    term_key: str = Field(min_length=1, max_length=200)
    difficulty: float = Field(ge=0.0, le=1.0)

    @field_validator("term_key")
    @classmethod
    def strip_term_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LessonCalibration(StrictModel):
    sequence: int = Field(ge=1)
    probes: list[CalibrationProbe] = Field(min_length=8, max_length=20)

    @model_validator(mode="after")
    def validate_unique_probes(self) -> LessonCalibration:
        keys = [probe.term_key for probe in self.probes]
        if len(keys) != len(set(keys)):
            raise ValueError("calibration probe term keys must be unique")
        return self


class LessonDocument(StrictModel):
    schema_version: Literal[1] = LESSON_SCHEMA_VERSION
    key: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    title_sentence: LessonSentence | None = None
    learning_language: LanguageTag = Field(description=LANGUAGE_TAG_DESCRIPTION)
    translation_language: LanguageTag = Field(description=LANGUAGE_TAG_DESCRIPTION)
    topic: str | None = Field(default=None, max_length=200)
    level: CefrLevel = "A1"
    difficulty: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Fine-grained difficulty within and across CEFR bands.",
    )
    blocks: list[LessonBlock] = Field(min_length=1)
    target_term_keys: list[str] = Field(default_factory=list)
    calibration: LessonCalibration | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "key",
        "title",
        "topic",
    )
    @classmethod
    def strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_structure(self) -> LessonDocument:
        block_keys: set[str] = set()
        sentence_keys: set[str] = set()
        terms: dict[str, LessonTerm] = {}

        if self.title_sentence is not None:
            if self.title_sentence.text != self.title:
                raise ValueError("title_sentence text must exactly equal title")
            sentence_keys.add(self.title_sentence.key)
            _collect_sentence_terms(self.title_sentence, terms)

        for block in self.blocks:
            if block.key in block_keys:
                raise ValueError(f"duplicate block key: {block.key}")
            block_keys.add(block.key)

            for sentence in block.sentences:
                if sentence.key in sentence_keys:
                    raise ValueError(f"duplicate sentence key: {sentence.key}")
                sentence_keys.add(sentence.key)
                _collect_sentence_terms(sentence, terms)

        if len(self.target_term_keys) != len(set(self.target_term_keys)):
            raise ValueError("target_term_keys must be unique")
        unknown_targets = set(self.target_term_keys) - terms.keys()
        if unknown_targets:
            unknown = ", ".join(sorted(unknown_targets))
            raise ValueError(f"target_term_keys contains unknown terms: {unknown}")

        _validate_calibration_structure(self.calibration, self.blocks, self.title_sentence)
        _validate_grammar_structure(
            self.learning_language,
            self.blocks,
            self.title_sentence,
        )

        return self

    def term_catalog(self) -> dict[str, LessonTerm]:
        terms = {
            run.term.key: run.term
            for block in self.blocks
            for sentence in block.sentences
            for run in sentence.runs
            if run.term is not None
        }
        if self.title_sentence is not None:
            terms.update(
                {run.term.key: run.term for run in self.title_sentence.runs if run.term is not None}
            )
        return terms

    def sentence_terms(self) -> dict[str, set[str]]:
        terms = {
            sentence.key: {run.term.key for run in sentence.runs if run.term is not None}
            for block in self.blocks
            for sentence in block.sentences
        }
        if self.title_sentence is not None:
            terms[self.title_sentence.key] = {
                run.term.key for run in self.title_sentence.runs if run.term is not None
            }
        return terms

    def calibration_probe_sentences(self) -> dict[str, str]:
        if self.calibration is None:
            return {}
        probe_keys = {probe.term_key for probe in self.calibration.probes}
        sentences = {
            run.term.key: sentence.key
            for block in self.blocks
            for sentence in block.sentences
            for run in sentence.runs
            if run.term is not None and run.term.key in probe_keys
        }
        if self.title_sentence is not None:
            sentences.update(
                {
                    run.term.key: self.title_sentence.key
                    for run in self.title_sentence.runs
                    if run.term is not None and run.term.key in probe_keys
                }
            )
        return sentences

    def grammar_occurrences(self) -> dict[str, GrammarOccurrence]:
        return {
            occurrence.key: occurrence
            for sentence in _lesson_sentences(self.blocks, self.title_sentence)
            for occurrence in sentence.grammar
        }

    def grammar_occurrence_sentences(self) -> dict[str, str]:
        return {
            occurrence.key: sentence.key
            for sentence in _lesson_sentences(self.blocks, self.title_sentence)
            for occurrence in sentence.grammar
        }

    def grammar_construction_keys(self) -> set[str]:
        return {
            occurrence.construction_key
            for sentence in _lesson_sentences(self.blocks, self.title_sentence)
            for occurrence in sentence.grammar
        }


def _validate_calibration_structure(
    calibration: LessonCalibration | None,
    blocks: Sequence[LessonBlock],
    title_sentence: LessonSentence | None = None,
) -> None:
    if calibration is None:
        return
    occurrences: dict[str, int] = {}
    sentences = [
        *(sentence for block in blocks for sentence in block.sentences),
        *([title_sentence] if title_sentence is not None else []),
    ]
    for sentence in sentences:
        for run in sentence.runs:
            if run.term is not None:
                occurrences[run.term.key] = occurrences.get(run.term.key, 0) + 1
    unknown = [probe.term_key for probe in calibration.probes if probe.term_key not in occurrences]
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"calibration contains unknown probe terms: {names}")
    repeated = [probe.term_key for probe in calibration.probes if occurrences[probe.term_key] != 1]
    if repeated:
        names = ", ".join(sorted(repeated))
        raise ValueError(f"calibration probe terms must occur exactly once: {names}")


def _lesson_sentences(
    blocks: Sequence[LessonBlock],
    title_sentence: LessonSentence | None = None,
) -> list[LessonSentence]:
    return [
        *([title_sentence] if title_sentence is not None else []),
        *(sentence for block in blocks for sentence in block.sentences),
    ]


def _validate_unique_grammar_occurrence_keys(sentences: Sequence[LessonSentence]) -> None:
    occurrence_keys = [occurrence.key for sentence in sentences for occurrence in sentence.grammar]
    if len(occurrence_keys) != len(set(occurrence_keys)):
        raise ValueError("grammar occurrence keys must be unique across a lesson")


def _validate_grammar_structure(
    learning_language: str,
    blocks: Sequence[LessonBlock],
    title_sentence: LessonSentence | None = None,
) -> None:
    sentences = _lesson_sentences(blocks, title_sentence)
    _validate_unique_grammar_occurrence_keys(sentences)
    construction_keys = {
        occurrence.construction_key for sentence in sentences for occurrence in sentence.grammar
    }
    if not construction_keys:
        return
    try:
        catalog = grammar_catalog(learning_language)
    except LookupError as error:
        raise ValueError(str(error)) from error
    unknown = construction_keys - {construction.key for construction in catalog.constructions}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(
            f"grammar constructions do not belong to learning language {learning_language}: {names}"
        )
