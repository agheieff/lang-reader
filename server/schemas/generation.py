"""The annotation stages and their request/result contracts: lexical, translation, grammar."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from server.schemas.agent import (
    AgentBrief,
    AgentTermBrief,
    ContentHistoryItem,
    ContentMove,
)
from server.schemas.base import (
    CefrLevel,
    FeedbackTag,
    GenerationRequestKind,
    LanguageTag,
    StrictModel,
    SubjectDomain,
    is_proper_noun_pos,
)
from server.schemas.lesson import (
    LessonRun,
    LessonTerm,
)
from server.schemas.prose import (
    CallbackLessonDraft,
    CallbackLessonRun,
    GeneratedLessonDraft,
    _validate_generated_run_coverage,
)
from server.schemas.views import (
    AgentGrammarCatalogEntry,
)


class CalibrationGenerationBrief(StrictModel):
    sequence: int = Field(ge=1)
    target_difficulty: float = Field(ge=0.0, le=1.0)
    probe_difficulties: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        min_length=8, max_length=20
    )
    excluded_term_keys: list[str]


class GenerationTargetPolicy(StrictModel):
    preferred_count: int = Field(ge=0, le=16)
    max_count: int = Field(ge=0, le=16)
    target_text_length: int = Field(ge=1, le=10_000)
    allow_zero: bool = True
    candidate_term_keys: list[str] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_policy(self) -> GenerationTargetPolicy:
        if self.preferred_count > self.max_count:
            raise ValueError("preferred_count must not exceed max_count")
        if len(self.candidate_term_keys) != len(set(self.candidate_term_keys)):
            raise ValueError("candidate_term_keys must be unique")
        return self


class GenerationGrammarPolicy(StrictModel):
    preferred_count: int = Field(default=0, ge=0, le=4)
    max_count: int = Field(default=0, ge=0, le=4)
    offered_keys: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def validate_policy(self) -> GenerationGrammarPolicy:
        if self.preferred_count > self.max_count:
            raise ValueError("grammar preferred_count must not exceed max_count")
        if self.max_count > len(self.offered_keys):
            raise ValueError("grammar max_count must not exceed the offered pool")
        if len(self.offered_keys) != len(set(self.offered_keys)):
            raise ValueError("grammar offered_keys must be unique")
        return self


class GenerationContentPlan(StrictModel):
    archetype: str = Field(min_length=1, max_length=100)
    discourse_form: str = Field(min_length=1, max_length=200)
    perspective: str = Field(min_length=1, max_length=200)
    concrete_seed: str = Field(min_length=1, max_length=500)
    progression: str = Field(min_length=1, max_length=500)
    ending_shape: str = Field(min_length=1, max_length=300)
    avoid_patterns: list[str] = Field(default_factory=list, max_length=8)
    move: ContentMove = "favourite"
    move_instruction: str = Field(default="", max_length=1_000)
    avoid_domains: list[SubjectDomain] = Field(default_factory=list, max_length=24)


class GenerationFailureContext(StrictModel):
    task_id: int = Field(gt=0)
    attempt: int = Field(gt=0)
    error: str = Field(min_length=1, max_length=4000)


class ProseLessonSentence(StrictModel):
    """Frozen learning-language text before lexical annotation."""

    key: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ProseLessonBlock(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    sentences: list[ProseLessonSentence] = Field(min_length=1, max_length=20)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ProseLessonDraft(StrictModel):
    """Complete prose whose text and stable keys remain immutable in downstream tasks."""

    title: str = Field(min_length=1, max_length=300)
    title_sentence: ProseLessonSentence
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
    blocks: list[ProseLessonBlock] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_structure(self) -> ProseLessonDraft:
        if self.title_sentence.text != self.title:
            raise ValueError("prose title_sentence text must exactly equal title")
        sentences = [
            self.title_sentence,
            *(sentence for block in self.blocks for sentence in block.sentences),
        ]
        keys = [sentence.key for sentence in sentences]
        if len(keys) != len(set(keys)):
            raise ValueError("prose sentence keys must be unique within a lesson")
        if len(sentences) > 41:
            raise ValueError("prose lesson must contain at most 40 body sentences")
        block_keys = [block.key for block in self.blocks]
        if len(block_keys) != len(set(block_keys)):
            raise ValueError("prose block keys must be unique within a lesson")
        return self


class GenerationProseResult(StrictModel):
    schema_version: Literal[1]
    lessons: list[ProseLessonDraft] = Field(min_length=1, max_length=10)


class LexicalLessonSentence(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    runs: list[CallbackLessonRun] = Field(min_length=1, max_length=100)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LexicalLessonBlock(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    sentences: list[LexicalLessonSentence] = Field(min_length=1, max_length=20)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LexicalLessonDraft(StrictModel):
    """Term catalog and token boundaries without translations or grammar."""

    terms: list[LessonTerm] = Field(min_length=1, max_length=4100)
    title_sentence: LexicalLessonSentence
    blocks: list[LexicalLessonBlock] = Field(min_length=1, max_length=12)
    target_term_keys: list[str] = Field(default_factory=list, max_length=16)

    def term_catalog(self) -> dict[str, LessonTerm]:
        catalog: dict[str, LessonTerm] = {}
        for term in self.terms:
            if term.key in catalog:
                raise ValueError(f"lexical term catalog defines key more than once: {term.key}")
            if term.frequency_rank is None and not is_proper_noun_pos(term.pos):
                raise ValueError(
                    "lexical non-name catalog terms require an approximate corpus "
                    f"frequency_rank: {term.key}"
                )
            catalog[term.key] = term
        return catalog

    @model_validator(mode="after")
    def validate_structure(self) -> LexicalLessonDraft:
        catalog = self.term_catalog()
        block_keys = [block.key for block in self.blocks]
        if len(block_keys) != len(set(block_keys)):
            raise ValueError("lexical block keys must be unique within a lesson")
        sentences = [
            self.title_sentence,
            *(sentence for block in self.blocks for sentence in block.sentences),
        ]
        sentence_keys = [sentence.key for sentence in sentences]
        if len(sentence_keys) != len(set(sentence_keys)):
            raise ValueError("lexical sentence keys must be unique within a lesson")
        for sentence in sentences:
            for run_index, run in enumerate(sentence.runs):
                if run.term_key is not None and run.term_key not in catalog:
                    raise ValueError(
                        f"lexical run references unknown term key {run.term_key!r} at "
                        f"sentence {sentence.key!r}, run {run_index}, surface {run.text!r}"
                    )
        if len(self.target_term_keys) != len(set(self.target_term_keys)):
            raise ValueError("lexical target_term_keys must be unique")
        unknown_targets = set(self.target_term_keys) - catalog.keys()
        if unknown_targets:
            names = ", ".join(sorted(unknown_targets))
            raise ValueError(f"lexical target_term_keys contains unknown terms: {names}")
        return self


class GenerationLexicalResult(StrictModel):
    schema_version: Literal[1]
    lessons: list[LexicalLessonDraft] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_stable_terms(self) -> GenerationLexicalResult:
        terms: dict[str, LessonTerm] = {}
        for lesson in self.lessons:
            for term in lesson.terms:
                previous = terms.get(term.key)
                if previous is not None and previous != term:
                    raise ValueError(
                        f"term key has conflicting lexical definitions across lessons: {term.key}"
                    )
                terms[term.key] = term
        return self


class SentenceTranslationDraft(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    translation: str = Field(min_length=1, max_length=20_000)

    @field_validator("key", "translation")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class TranslationLessonDraft(StrictModel):
    sentences: list[SentenceTranslationDraft] = Field(min_length=1, max_length=41)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> TranslationLessonDraft:
        keys = [sentence.key for sentence in self.sentences]
        if len(keys) != len(set(keys)):
            raise ValueError("translation sentence keys must be unique within a lesson")
        return self


class GenerationTranslationResult(StrictModel):
    schema_version: Literal[1]
    lessons: list[TranslationLessonDraft] = Field(min_length=1, max_length=10)


class CallbackGrammarOccurrence(StrictModel):
    """A provider-authored range; the host assigns its stable occurrence key."""

    construction_key: str = Field(min_length=1, max_length=100)
    run_start: int = Field(ge=0)
    run_end: int = Field(ge=1)
    note: str | None = Field(default=None, max_length=500)

    @field_validator("construction_key")
    @classmethod
    def strip_construction_key(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("note")
    @classmethod
    def strip_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_range(self) -> CallbackGrammarOccurrence:
        if self.run_end <= self.run_start:
            raise ValueError("grammar occurrence run_end must be greater than run_start")
        return self


class GrammarSentenceDraft(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    occurrences: list[CallbackGrammarOccurrence] = Field(default_factory=list, max_length=20)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_unique_occurrences(self) -> GrammarSentenceDraft:
        identities = [
            (
                occurrence.construction_key,
                occurrence.run_start,
                occurrence.run_end,
            )
            for occurrence in self.occurrences
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("grammar occurrences must not duplicate a construction and range")
        return self


class GrammarLessonDraft(StrictModel):
    sentences: list[GrammarSentenceDraft] = Field(min_length=1, max_length=41)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> GrammarLessonDraft:
        keys = [sentence.key for sentence in self.sentences]
        if len(keys) != len(set(keys)):
            raise ValueError("grammar sentence keys must be unique within a lesson")
        return self


class GenerationGrammarResult(StrictModel):
    schema_version: Literal[1]
    lessons: list[GrammarLessonDraft] = Field(min_length=1, max_length=10)


class TokenizedLessonSentence(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    runs: list[str] = Field(min_length=1, max_length=100)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("runs")
    @classmethod
    def validate_runs(cls, value: list[str]) -> list[str]:
        if any(not run for run in value):
            raise ValueError("tokenized lesson runs must not be empty")
        return value


class TokenizedLessonBlock(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    sentences: list[TokenizedLessonSentence] = Field(min_length=1, max_length=20)


class TokenizedLessonDraft(StrictModel):
    title_sentence: TokenizedLessonSentence
    blocks: list[TokenizedLessonBlock] = Field(min_length=1, max_length=12)


class GenerationStageRequestBase(StrictModel):
    schema_version: Literal[3] = 3
    stage: Literal["lexical", "translation", "grammar"]
    task: Literal["lexical", "translation", "grammar"]
    job_id: str = Field(min_length=3, max_length=100)
    task_id: int = Field(gt=0)
    attempt: int = Field(gt=0)
    lesson_count: int = Field(ge=1, le=10)
    profile_key: str = Field(min_length=1, max_length=100)
    workspace_path: str = Field(min_length=1, max_length=1000)
    profile_fingerprint: str = Field(min_length=64, max_length=64)
    learning_language: LanguageTag
    translation_language: LanguageTag
    previous_failures: list[GenerationFailureContext] = Field(default_factory=list, max_length=4)
    instructions: str

    @model_validator(mode="after")
    def validate_stage_task(self) -> GenerationStageRequestBase:
        if self.stage != self.task:
            raise ValueError("staged generation request stage and task must match")
        return self


class GenerationLexicalUnitResult(StrictModel):
    """Sentence-local lexical output; term keys are scoped to this unit until host assembly."""

    schema_version: Literal[1]
    unit_id: str = Field(min_length=1, max_length=200)
    key: str = Field(min_length=1, max_length=200)
    terms: list[LessonTerm] = Field(max_length=200)
    runs: list[CallbackLessonRun] = Field(min_length=1, max_length=100)

    @field_validator("unit_id", "key")
    @classmethod
    def strip_identifiers(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_catalog_and_runs(self) -> GenerationLexicalUnitResult:
        catalog: dict[str, LessonTerm] = {}
        for term in self.terms:
            if term.key in catalog:
                raise ValueError(f"lexical unit defines term key more than once: {term.key}")
            if term.frequency_rank is None and not is_proper_noun_pos(term.pos):
                raise ValueError(
                    "lexical unit non-name terms require an approximate corpus "
                    f"frequency_rank: {term.key}"
                )
            catalog[term.key] = term

        for run_index, run in enumerate(self.runs):
            resolved_term = catalog.get(run.term_key) if run.term_key is not None else None
            if run.term_key is not None and resolved_term is None:
                raise ValueError(
                    f"lexical unit run references unknown term key {run.term_key!r} at "
                    f"run {run_index}, surface {run.text!r}"
                )
            _validate_generated_run_coverage(
                LessonRun(
                    text=run.text,
                    term=resolved_term,
                    pronunciation=run.pronunciation,
                )
            )
        return self


class GenerationLexicalUnitRequest(GenerationStageRequestBase):
    stage: Literal["lexical"] = "lexical"
    task: Literal["lexical"] = "lexical"
    request_kind: Literal["sentence"] = "sentence"
    unit_id: str = Field(min_length=1, max_length=200)
    lesson_index: int = Field(ge=0, le=9)
    sentence_index: int = Field(ge=0, le=40)
    unit_attempt: int = Field(default=1, ge=1, le=2)
    is_title: bool
    frozen_sentence: ProseLessonSentence
    context_sentences: list[ProseLessonSentence] = Field(default_factory=list, max_length=41)
    known_terms: list[LessonTerm] = Field(default_factory=list, max_length=256)
    language_guidance: list[str] = Field(default_factory=list, max_length=32)
    repair_result: GenerationLexicalUnitResult | None = None

    @field_validator("unit_id")
    @classmethod
    def strip_unit_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("language_guidance")
    @classmethod
    def validate_language_guidance(cls, value: list[str]) -> list[str]:
        guidance = [item.strip() for item in value]
        if any(not item for item in guidance):
            raise ValueError("language guidance must not contain blank entries")
        if any(len(item) > 4_000 for item in guidance):
            raise ValueError("language guidance entries must contain at most 4000 characters")
        return guidance

    @model_validator(mode="after")
    def validate_unit_context(self) -> GenerationLexicalUnitRequest:
        if self.lesson_index >= self.lesson_count:
            raise ValueError("lexical unit lesson_index is outside lesson_count")
        context_by_key: dict[str, ProseLessonSentence] = {}
        for sentence in self.context_sentences:
            previous = context_by_key.get(sentence.key)
            if previous is not None:
                raise ValueError(f"lexical unit repeats context sentence key: {sentence.key}")
            context_by_key[sentence.key] = sentence
        matching_context = context_by_key.get(self.frozen_sentence.key)
        if matching_context is not None and matching_context != self.frozen_sentence:
            raise ValueError("lexical unit context conflicts with its frozen sentence")

        known_keys: set[str] = set()
        for term in self.known_terms:
            if term.key in known_keys:
                raise ValueError(f"lexical unit repeats known term key: {term.key}")
            known_keys.add(term.key)

        if self.repair_result is not None:
            if self.attempt <= 1 and self.unit_attempt <= 1:
                raise ValueError("lexical unit repair_result is valid only on a task or unit retry")
            if self.repair_result.unit_id != self.unit_id:
                raise ValueError("lexical unit repair_result belongs to a different unit")
            if self.repair_result.key != self.frozen_sentence.key:
                raise ValueError("lexical unit repair_result has a different sentence key")
        return self


class LexicalBatchUnit(StrictModel):
    unit_id: str = Field(min_length=1, max_length=200)
    lesson_index: int = Field(ge=0, le=9)
    sentence_index: int = Field(ge=0, le=40)
    is_title: bool
    frozen_sentence: ProseLessonSentence


MAX_LEXICAL_BATCH_UNITS = 16


class GenerationLexicalBatchRequest(GenerationStageRequestBase):
    """Several frozen sentences tokenized in one callback; each result stays sentence-local."""

    stage: Literal["lexical"] = "lexical"
    task: Literal["lexical"] = "lexical"
    request_kind: Literal["sentence_batch"] = "sentence_batch"
    units: list[LexicalBatchUnit] = Field(min_length=1, max_length=MAX_LEXICAL_BATCH_UNITS)
    context_sentences: list[ProseLessonSentence] = Field(default_factory=list, max_length=41)
    known_terms: list[LessonTerm] = Field(default_factory=list, max_length=256)
    language_guidance: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_batch(self) -> GenerationLexicalBatchRequest:
        unit_ids = [unit.unit_id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("lexical batch repeats a unit id")
        if any(unit.lesson_index >= self.lesson_count for unit in self.units):
            raise ValueError("lexical batch lesson_index is outside lesson_count")
        known_keys = [term.key for term in self.known_terms]
        if len(known_keys) != len(set(known_keys)):
            raise ValueError("lexical batch repeats a known term key")
        return self


class GenerationLexicalBatchResult(StrictModel):
    """Response shape for a batch; the host validates each unit independently."""

    schema_version: Literal[1]
    units: list[GenerationLexicalUnitResult] = Field(
        min_length=1, max_length=MAX_LEXICAL_BATCH_UNITS
    )


class PreferenceUpdateRequest(StrictModel):
    """Ask the agent to maintain the learner's reading-preference notes."""

    schema_version: Literal[1] = 1
    stage: Literal["preferences"] = "preferences"
    task: Literal["prose"] = "prose"
    job_id: str = Field(min_length=3, max_length=100)
    task_id: int = Field(ge=0)
    profile_key: str = Field(min_length=1, max_length=100)
    workspace_path: str = Field(min_length=1, max_length=1000)
    learning_language: LanguageTag
    current_preferences: str = Field(max_length=4_000)
    user_messages: list[str] = Field(default_factory=list, max_length=20)
    content_history: list[ContentHistoryItem] = Field(default_factory=list, max_length=40)
    instructions: str


class PreferenceUpdateResult(StrictModel):
    schema_version: Literal[1]
    preferences: str = Field(min_length=1, max_length=4_000)
    reason: str = Field(min_length=1, max_length=500)


class LexicalConflictContext(StrictModel):
    sentence_key: str = Field(min_length=1, max_length=200)
    surface: str = Field(min_length=1, max_length=500)
    sentence_text: str = Field(min_length=1, max_length=20_000)

    @field_validator("sentence_key", "surface", "sentence_text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LexicalConflictCandidate(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=200)
    term: LessonTerm
    stored: bool
    contexts: list[LexicalConflictContext] = Field(default_factory=list, max_length=41)

    @field_validator("candidate_id")
    @classmethod
    def strip_candidate_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_contexts(self) -> LexicalConflictCandidate:
        identities = [
            (context.sentence_key, context.surface, context.sentence_text)
            for context in self.contexts
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("lexical conflict candidate contexts must be unique")
        if not self.stored and not self.contexts:
            raise ValueError("generated lexical conflict candidates require context")
        return self


class LexicalConflictGroup(StrictModel):
    group_id: str = Field(min_length=1, max_length=200)
    candidates: list[LexicalConflictCandidate] = Field(min_length=2, max_length=16)

    @field_validator("group_id")
    @classmethod
    def strip_group_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> LexicalConflictGroup:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("lexical conflict candidate IDs must be unique within a group")
        return self


class GenerationLexicalConflictRequest(GenerationStageRequestBase):
    stage: Literal["lexical"] = "lexical"
    task: Literal["lexical"] = "lexical"
    request_kind: Literal["conflicts"] = "conflicts"
    groups: list[LexicalConflictGroup] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_group_and_candidate_ids(self) -> GenerationLexicalConflictRequest:
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("lexical conflict group IDs must be unique")
        candidate_ids = [
            candidate.candidate_id for group in self.groups for candidate in group.candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("lexical conflict candidate IDs must be globally unique")
        return self


class LexicalConflictResolution(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=200)
    canonical_candidate_id: str = Field(min_length=1, max_length=200)

    @field_validator("candidate_id", "canonical_candidate_id")
    @classmethod
    def strip_candidate_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class GenerationLexicalConflictResult(StrictModel):
    schema_version: Literal[1]
    resolutions: list[LexicalConflictResolution] = Field(min_length=1, max_length=1_024)

    @model_validator(mode="after")
    def validate_resolutions(self) -> GenerationLexicalConflictResult:
        candidate_ids = [resolution.candidate_id for resolution in self.resolutions]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("lexical conflict result resolves a candidate more than once")
        return self


class GenerationLexicalRequest(GenerationStageRequestBase):
    stage: Literal["lexical"] = "lexical"
    task: Literal["lexical"] = "lexical"
    frozen_lessons: list[ProseLessonDraft] = Field(min_length=1, max_length=10)
    known_terms: list[AgentTermBrief] = Field(default_factory=list, max_length=64)
    target_policy: GenerationTargetPolicy
    language_guidance: list[str]
    repair_lessons: list[LexicalLessonDraft] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_lesson_count(self) -> GenerationLexicalRequest:
        if len(self.frozen_lessons) != self.lesson_count:
            raise ValueError("lexical request requires exactly lesson_count frozen lessons")
        if self.repair_lessons and self.attempt <= 1:
            raise ValueError("lexical repair_lessons are valid only on a retry")
        return self


class GenerationTranslationRequest(GenerationStageRequestBase):
    stage: Literal["translation"] = "translation"
    task: Literal["translation"] = "translation"
    frozen_lessons: list[ProseLessonDraft] = Field(min_length=1, max_length=10)
    repair_lessons: list[TranslationLessonDraft] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_lesson_count(self) -> GenerationTranslationRequest:
        if len(self.frozen_lessons) != self.lesson_count:
            raise ValueError("translation request requires exactly lesson_count frozen lessons")
        if self.repair_lessons and self.attempt <= 1:
            raise ValueError("translation repair_lessons are valid only on a retry")
        return self


class GenerationGrammarRequest(GenerationStageRequestBase):
    stage: Literal["grammar"] = "grammar"
    task: Literal["grammar"] = "grammar"
    tokenized_lessons: list[TokenizedLessonDraft] = Field(min_length=1, max_length=10)
    grammar_catalog: list[AgentGrammarCatalogEntry] = Field(default_factory=list)
    repair_lessons: list[GrammarLessonDraft] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_lesson_count(self) -> GenerationGrammarRequest:
        if len(self.tokenized_lessons) != self.lesson_count:
            raise ValueError("grammar request requires exactly lesson_count tokenized lessons")
        if self.repair_lessons and self.attempt <= 1:
            raise ValueError("grammar repair_lessons are valid only on a retry")
        return self


class GenerationCallbackRequest(StrictModel):
    schema_version: Literal[3] = 3
    stage: Literal["legacy", "prose"] = "legacy"
    task: Literal["complete", "prose"] = "complete"
    job_id: str = Field(min_length=3, max_length=100)
    task_id: int = Field(gt=0)
    attempt: int = Field(gt=0)
    lesson_count: int = Field(ge=1, le=10)
    lexical_coverage: Literal["all_lexical_tokens"] = "all_lexical_tokens"
    profile_key: str = Field(min_length=1, max_length=100)
    workspace_path: str = Field(min_length=1, max_length=1000)
    profile_fingerprint: str = Field(min_length=64, max_length=64)
    brief: AgentBrief
    target_policy: GenerationTargetPolicy
    grammar_policy: GenerationGrammarPolicy = Field(default_factory=GenerationGrammarPolicy)
    content_plan: GenerationContentPlan | None = None
    calibration: CalibrationGenerationBrief | None = None
    request_kind: GenerationRequestKind = "queue_fill"
    requested_topic: str | None = Field(default=None, min_length=1, max_length=200)
    latest_feedback: list[FeedbackTag] = Field(default_factory=list, max_length=8)
    previous_failures: list[GenerationFailureContext] = Field(default_factory=list, max_length=4)
    repair_lessons: list[CallbackLessonDraft] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Rejected compact drafts from this task's immediately preceding attempt. On retry, "
            "repair their annotations in place instead of replacing their content."
        ),
    )
    instructions: str

    @model_validator(mode="after")
    def validate_target_candidates(self) -> GenerationCallbackRequest:
        priority_keys = {term.key for term in self.brief.priority_terms}
        unknown = set(self.target_policy.candidate_term_keys) - priority_keys
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"target policy contains terms outside the priority pool: {names}")
        grammar_keys = {construction.key for construction in self.brief.priority_grammar}
        unknown_grammar = set(self.grammar_policy.offered_keys) - grammar_keys
        if unknown_grammar:
            names = ", ".join(sorted(unknown_grammar))
            raise ValueError(
                f"grammar policy contains constructions outside the priority pool: {names}"
            )
        if self.request_kind == "topic_request" and self.requested_topic is None:
            raise ValueError("topic requests require requested_topic")
        if self.request_kind == "queue_fill" and self.requested_topic is not None:
            raise ValueError("queue-fill requests must not include requested_topic")
        if self.calibration is None and self.content_plan is None:
            raise ValueError("ordinary generation requests require content_plan")
        if self.calibration is not None and self.content_plan is not None:
            raise ValueError("calibration generation requests must not include content_plan")
        if self.calibration is not None and (
            self.grammar_policy.preferred_count
            or self.grammar_policy.max_count
            or self.grammar_policy.offered_keys
        ):
            raise ValueError("calibration generation requests must not offer priority grammar")
        if self.repair_lessons and self.attempt <= 1:
            raise ValueError("repair_lessons are valid only on a callback retry")
        if self.calibration is not None and self.stage != "legacy":
            raise ValueError("calibration generation uses the legacy one-pass stage")
        if self.stage == "prose" and self.task != "prose":
            raise ValueError("the prose stage requires the prose task")
        if self.stage == "legacy" and self.task != "complete":
            raise ValueError("the legacy stage requires the complete task")
        return self


class GenerationCallbackResult(StrictModel):
    schema_version: Literal[1]
    lessons: list[CallbackLessonDraft] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_stable_terms(self) -> GenerationCallbackResult:
        terms: dict[str, LessonTerm] = {}
        for lesson in self.lessons:
            for term in lesson.terms:
                previous = terms.get(term.key)
                if previous is not None and previous != term:
                    raise ValueError(
                        f"term key has conflicting definitions across generated lessons: {term.key}"
                    )
                terms[term.key] = term
        return self

    def expanded_lessons(self) -> list[GeneratedLessonDraft]:
        return [lesson.expand() for lesson in self.lessons]
