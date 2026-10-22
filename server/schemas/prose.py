"""The first generation stages: complete lessons and frozen prose."""

from __future__ import annotations

import unicodedata
from typing import Any

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from server.schemas.base import (
    CefrLevel,
    StrictModel,
    SubjectDomain,
    is_proper_noun_pos,
)
from server.schemas.lesson import (
    GrammarOccurrence,
    LessonBlock,
    LessonCalibration,
    LessonRun,
    LessonSentence,
    LessonTerm,
    _collect_sentence_terms,
    _validate_calibration_structure,
    _validate_unique_grammar_occurrence_keys,
)


class GeneratedLessonBlock(LessonBlock):
    sentences: list[LessonSentence] = Field(min_length=1, max_length=20)


class GeneratedLessonDraft(StrictModel):
    title: str = Field(min_length=1, max_length=300)
    title_sentence: LessonSentence = Field(
        description=(
            "Fully annotated title whose text exactly equals title and includes its translation."
        )
    )
    topic: str | None = Field(max_length=200)
    content_angle: str | None = Field(
        default=None,
        max_length=160,
        description="A few words naming the specific subject and angle chosen for this text.",
    )
    hypothesis: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "For a variation or new-subject text: the preference question it tests, phrased for "
            "the learner, e.g. 'Do you enjoy maritime history?'. Otherwise null."
        ),
    )
    subject_domain: SubjectDomain | None = Field(
        default=None,
        description=(
            "The one broad subject area of this text. Never one listed in "
            "content_plan.avoid_domains."
        ),
    )
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    blocks: list[GeneratedLessonBlock] = Field(
        min_length=1,
        max_length=12,
        description=(
            "Fully clickable text: every lexical token is a term-bearing run; plain runs contain "
            "only whitespace or punctuation."
        ),
    )
    target_term_keys: list[str] = Field(
        max_length=16,
        description=(
            "Advisory target choices. The host reconciles these against the offered candidates "
            "and terms actually present in the lesson."
        ),
    )
    calibration: LessonCalibration | None = Field(
        default=None,
        description="Optional calibration-probe subset, independent of target_term_keys.",
    )

    @model_validator(mode="after")
    def validate_targets(self) -> GeneratedLessonDraft:
        terms: dict[str, LessonTerm] = {}
        if self.title_sentence.text != self.title:
            raise ValueError("title_sentence text must exactly equal title")
        body_sentences = [sentence for block in self.blocks for sentence in block.sentences]
        sentences = [self.title_sentence, *body_sentences]
        sentence_keys = [sentence.key for sentence in sentences]
        if len(sentence_keys) != len(set(sentence_keys)):
            raise ValueError("generated sentence keys must be unique, including title_sentence")
        _validate_unique_grammar_occurrence_keys(sentences)
        for sentence in sentences:
            for run in sentence.runs:
                _validate_generated_run_coverage(run)
                if (
                    run.term is not None
                    and run.term.frequency_rank is None
                    and not is_proper_noun_pos(run.term.pos)
                ):
                    raise ValueError(
                        "generated non-name terms require an approximate corpus frequency_rank"
                    )
            _collect_sentence_terms(sentence, terms, generated=True)
        if len(self.target_term_keys) != len(set(self.target_term_keys)):
            raise ValueError("target_term_keys must be unique")
        if len(sentences) > 40:
            raise ValueError("lesson draft must contain at most 40 sentences")
        if any(len(block.sentences) > 20 for block in self.blocks):
            raise ValueError("lesson draft blocks must contain at most 20 sentences")
        if any(len(sentence.runs) > 100 for sentence in sentences):
            raise ValueError("lesson draft sentences must contain at most 100 runs")
        if any(sentence.translation is None for sentence in sentences):
            raise ValueError("every generated sentence must include a translation")
        _validate_calibration_structure(self.calibration, self.blocks, self.title_sentence)
        return self


class CallbackLessonRun(StrictModel):
    """Compact callback run; canonical term metadata lives in the lesson catalog."""

    text: str = Field(min_length=1)
    term_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Key in this lesson's terms catalog, or null for whitespace/punctuation.",
    )
    pronunciation: str | None = Field(
        default=None,
        max_length=300,
        description="Optional contextual surface reading; dictionary reading belongs on the term.",
    )

    @field_validator("term_key", "pronunciation")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class CallbackLessonSentence(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    runs: list[CallbackLessonRun] = Field(min_length=1, max_length=100)
    translation: str
    grammar: list[GrammarOccurrence] = Field(default_factory=list, max_length=20)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class CallbackLessonBlock(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    sentences: list[CallbackLessonSentence] = Field(min_length=1, max_length=20)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class CallbackLessonDraft(StrictModel):
    """Token-efficient callback contract expanded before host validation and persistence."""

    title: str = Field(min_length=1, max_length=300)
    terms: list[LessonTerm] = Field(
        min_length=1,
        max_length=4100,
        description=(
            "Lesson-local term catalog. Define each key exactly once, then reference it from runs."
        ),
    )
    title_sentence: CallbackLessonSentence = Field(
        description=(
            "Fully annotated title whose run text exactly equals title and includes its "
            "translation."
        )
    )
    topic: str | None = Field(max_length=200)
    content_angle: str | None = Field(default=None, max_length=160)
    hypothesis: str | None = Field(default=None, max_length=300)
    subject_domain: SubjectDomain | None = None
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    blocks: list[CallbackLessonBlock] = Field(min_length=1, max_length=12)
    target_term_keys: list[str] = Field(
        max_length=16,
        description=(
            "Advisory target choices. The host reconciles these against offered candidates and "
            "terms present in the body."
        ),
    )
    calibration: LessonCalibration | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def accept_expanded_python_drafts(cls, value: Any) -> Any:
        """Keep programmatic fixtures compatible while the emitted JSON schema stays compact."""

        if isinstance(value, GeneratedLessonDraft):
            value = value.model_dump(mode="python")
        if not isinstance(value, dict) or "terms" in value:
            return value
        if "title_sentence" not in value or "blocks" not in value:
            return value
        return _compact_expanded_lesson(value)

    def term_catalog(self) -> dict[str, LessonTerm]:
        catalog: dict[str, LessonTerm] = {}
        for term in self.terms:
            if term.key in catalog:
                raise ValueError(f"callback term catalog defines key more than once: {term.key}")
            if term.frequency_rank is None and not is_proper_noun_pos(term.pos):
                raise ValueError(
                    "callback non-name catalog terms require an approximate corpus "
                    f"frequency_rank: {term.key}"
                )
            catalog[term.key] = term
        return catalog

    def expand(self) -> GeneratedLessonDraft:
        catalog = self.term_catalog()

        def expand_sentence(sentence: CallbackLessonSentence) -> LessonSentence:
            runs: list[LessonRun] = []
            for run_index, run in enumerate(sentence.runs):
                term = None
                if run.term_key is not None:
                    term = catalog.get(run.term_key)
                    if term is None:
                        raise ValueError(
                            f"callback run references unknown term key {run.term_key!r} at "
                            f"sentence {sentence.key!r}, run {run_index}, surface {run.text!r}"
                        )
                runs.append(
                    LessonRun(
                        text=run.text,
                        term=term,
                        pronunciation=run.pronunciation,
                    )
                )
            return LessonSentence(
                key=sentence.key,
                runs=runs,
                translation=sentence.translation,
                grammar=sentence.grammar,
            )

        title_sentence = expand_sentence(self.title_sentence)
        blocks = [
            GeneratedLessonBlock(
                key=block.key,
                sentences=[expand_sentence(sentence) for sentence in block.sentences],
            )
            for block in self.blocks
        ]
        return GeneratedLessonDraft(
            title=self.title,
            title_sentence=title_sentence,
            topic=self.topic,
            content_angle=self.content_angle,
            hypothesis=self.hypothesis,
            subject_domain=self.subject_domain,
            level=self.level,
            difficulty=self.difficulty,
            blocks=blocks,
            target_term_keys=self.target_term_keys,
            calibration=self.calibration,
        )

    @model_validator(mode="after")
    def validate_expanded_lesson(self) -> CallbackLessonDraft:
        catalog = self.term_catalog()
        unresolved = any(
            run.term_key is not None and run.term_key not in catalog
            for sentence in [
                self.title_sentence,
                *(sentence for block in self.blocks for sentence in block.sentences),
            ]
            for run in sentence.runs
        )
        if not unresolved:
            self.expand()
        return self


def _compact_expanded_lesson(value: dict[str, Any]) -> dict[str, Any]:
    """Convert the old repeated-term Python shape without exposing it in callback JSON Schema."""

    compact = dict(value)
    catalog: dict[str, LessonTerm] = {}

    def compact_sentence(raw_sentence: Any) -> Any:
        if isinstance(raw_sentence, LessonSentence):
            sentence = raw_sentence.model_dump(mode="python")
        elif isinstance(raw_sentence, dict):
            sentence = dict(raw_sentence)
        else:
            return raw_sentence
        compact_runs: list[dict[str, Any]] = []
        for raw_run in sentence.get("runs", []):
            if isinstance(raw_run, LessonRun):
                run = raw_run.model_dump(mode="python")
            else:
                run = dict(raw_run)
            raw_term = run.pop("term", None)
            if raw_term is None:
                run.setdefault("term_key", None)
            else:
                term = LessonTerm.model_validate(raw_term)
                previous = catalog.get(term.key)
                if previous is not None and previous != term:
                    raise ValueError(f"term key has conflicting generated definitions: {term.key}")
                catalog[term.key] = term
                run["term_key"] = term.key
            compact_runs.append(run)
        sentence["runs"] = compact_runs
        return sentence

    compact["title_sentence"] = compact_sentence(compact["title_sentence"])
    compact_blocks: list[dict[str, Any]] = []
    for raw_block in compact["blocks"]:
        if isinstance(raw_block, LessonBlock):
            block = raw_block.model_dump(mode="python")
        else:
            block = dict(raw_block)
        block["sentences"] = [compact_sentence(sentence) for sentence in block.get("sentences", [])]
        compact_blocks.append(block)
    compact["blocks"] = compact_blocks
    compact["terms"] = list(catalog.values())
    return compact


_INTRA_TOKEN_PUNCTUATION = {"'", "’", "-", "‐", "‑"}


def _is_plain_generated_character(character: str) -> bool:
    category = unicodedata.category(character)
    return character.isspace() or category.startswith(("P", "Z"))


def _validate_generated_run_coverage(run: LessonRun) -> None:
    if run.term is None:
        if any(not _is_plain_generated_character(character) for character in run.text):
            raise ValueError(
                "generated plain runs may contain only whitespace and punctuation; annotate every "
                "lexical token with a term"
            )
        return

    has_lexical_character = any(
        unicodedata.category(character)[0] in {"L", "M", "N", "S"} for character in run.text
    )
    if not has_lexical_character:
        raise ValueError("generated term runs must contain a lexical token")
    for index, character in enumerate(run.text):
        if character.isspace() or unicodedata.category(character).startswith("Z"):
            raise ValueError("generated term runs must isolate one token without whitespace")
        if unicodedata.category(character).startswith("P") and (
            character not in _INTRA_TOKEN_PUNCTUATION or index == 0 or index == len(run.text) - 1
        ):
            raise ValueError(
                "generated term runs must keep surrounding punctuation in separate plain runs"
            )
