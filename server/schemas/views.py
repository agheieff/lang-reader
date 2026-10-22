"""What the reader reads back: profiles, texts, words, characters, statistics and grammar."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, get_args
from uuid import UUID

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from server.clock import utc_now
from server.schemas.base import (
    LANGUAGE_TAG_DESCRIPTION,
    CalibrationStatus,
    CefrLevel,
    EventType,
    FeedbackTag,
    GenerationRequestKind,
    GenerationTaskState,
    LanguageTag,
    LevelSource,
    StrictModel,
    TermKnowledgeBand,
    TextStatus,
)
from server.schemas.lesson import (
    LessonDocument,
)


class ProfileUpdate(StrictModel):
    learning_language: LanguageTag | None = Field(
        default=None, description=LANGUAGE_TAG_DESCRIPTION
    )
    translation_language: LanguageTag | None = Field(
        default=None, description=LANGUAGE_TAG_DESCRIPTION
    )
    level: CefrLevel | None = None
    difficulty: float | None = Field(default=None, ge=0.0, le=1.0)
    interests: list[str] | None = None
    preferences: dict[str, Any] | None = None

    @field_validator("learning_language", "translation_language")
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("interests")
    @classmethod
    def normalize_interests(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class ProficiencyView(StrictModel):
    source: LevelSource = "self_reported"
    status: CalibrationStatus = "unstarted"
    lower: float | None = Field(default=None, ge=0.0, le=1.0)
    upper: float | None = Field(default=None, ge=0.0, le=1.0)
    lower_level: CefrLevel | None = None
    upper_level: CefrLevel | None = None
    qualified_attempts: int = Field(default=0, ge=0)
    usable_probes: int = Field(default=0, ge=0)
    qualified_readings: int = Field(default=0, ge=0)


class ProfileView(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    interests: list[str]
    preferences: dict[str, Any]
    proficiency: ProficiencyView = Field(default_factory=ProficiencyView)


class WorkspaceSummary(StrictModel):
    profile_id: str
    label: str
    learning_language: LanguageTag
    translation_language: LanguageTag


class WorkspaceList(StrictModel):
    selected_profile_id: str
    profiles: list[WorkspaceSummary]


class InteractionIn(StrictModel):
    event_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    lesson_id: int = Field(gt=0)
    type: EventType
    occurred_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def strip_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_payload(self) -> InteractionIn:
        if self.type == "term.revealed":
            _require_string(self.payload, "term_key")
        elif self.type == "translation.revealed":
            scope = self.payload.get("scope")
            if scope not in {"sentence", "lesson", "grammar"}:
                raise ValueError(
                    "translation.revealed payload scope must be sentence, lesson, or grammar"
                )
            if scope in {"sentence", "grammar"}:
                _require_string(self.payload, "sentence_key")
            if scope == "grammar":
                _require_string(self.payload, "occurrence_key")
                _require_string(self.payload, "construction_key")
        elif self.type == "lesson.completed":
            _validate_completion_payload(self.payload)
        elif self.type == "lesson.rated":
            rating = self.payload.get("rating")
            if rating not in {-1, 1, None}:
                raise ValueError("lesson.rated payload rating must be -1, 1, or null")
            _validate_feedback(self.payload)
            if rating is None and not self.payload.get("feedback"):
                raise ValueError("lesson.rated payload must include a rating or feedback")
        return self


class InteractionBatch(StrictModel):
    events: list[InteractionIn] = Field(min_length=1)


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"payload.{key} must be a non-blank string")
    return value


def _validate_completion_payload(payload: dict[str, Any]) -> None:
    active_seconds = payload.get("active_seconds")
    completion_ratio = payload.get("completion_ratio")
    if active_seconds is not None and (
        not isinstance(active_seconds, (int, float)) or active_seconds < 0
    ):
        raise ValueError("payload.active_seconds must be non-negative")
    if completion_ratio is not None and (
        not isinstance(completion_ratio, (int, float)) or not 0 <= completion_ratio <= 1
    ):
        raise ValueError("payload.completion_ratio must be between 0 and 1")


def _validate_feedback(payload: dict[str, Any]) -> None:
    if "feedback" not in payload:
        return
    feedback = payload["feedback"]
    allowed = set(get_args(FeedbackTag))
    if not isinstance(feedback, list) or any(
        not isinstance(item, str) or item not in allowed for item in feedback
    ):
        raise ValueError("payload.feedback must be a list of known feedback tags")
    if len(feedback) != len(set(feedback)):
        raise ValueError("payload.feedback must contain unique tags")
    contradictions = (
        {"shorter", "longer"},
        {"easier", "more_challenging"},
        {"more_grammar", "less_grammar"},
        {"same_topic", "new_topic"},
    )
    if any(pair <= set(feedback) for pair in contradictions):
        raise ValueError("payload.feedback contains contradictory tags")


class EventRecordResult(StrictModel):
    accepted: int
    duplicates: int
    accepted_event_ids: list[str]


class ReaderProgress(StrictModel):
    session_id: str | None = None
    started: bool = False
    completed: bool = False
    rating: Literal[-1, 1] | None = None
    feedback: list[FeedbackTag] = Field(default_factory=list)
    revealed_term_keys: list[str] = Field(default_factory=list)
    revealed_sentence_keys: list[str] = Field(default_factory=list)
    revealed_grammar_occurrence_keys: list[str] = Field(default_factory=list)
    full_translation_revealed: bool = False


class AgentGrammarCatalogEntry(StrictModel):
    key: str
    label: str
    description: str
    category: str
    difficulty: float = Field(ge=0.0, le=1.0)
    generation_hint: str | None = None


class ReaderState(StrictModel):
    profile: ProfileView
    lesson_id: int | None = None
    lesson: LessonDocument | None = None
    progress: ReaderProgress = Field(default_factory=ReaderProgress)
    term_bands: dict[str, TermKnowledgeBand] = Field(default_factory=dict)
    grammar_catalog: list[AgentGrammarCatalogEntry] = Field(default_factory=list)
    comfortable_grammar_construction_keys: list[str] = Field(default_factory=list)
    generation_status: Literal["pending", "running", "failed"] | None = None


class TextView(StrictModel):
    id: int = Field(gt=0)
    key: str
    title: str
    topic: str | None
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    imported_at: datetime
    status: TextStatus
    queue_position: int | None = Field(default=None, ge=1)
    opened_at: datetime | None
    last_completed_at: datetime | None
    skipped_at: datetime | None = None
    session_count: int = Field(ge=0)
    completion_count: int = Field(ge=0)
    rating: Literal[-1, 1] | None
    lexical_token_count: int = Field(ge=0)
    # Predicted share of running words the learner currently knows.
    known_share: float | None = Field(default=None, ge=0.0, le=1.0)


class LessonQueueActionIn(StrictModel):
    action_id: UUID


class LessonQueueActionView(StrictModel):
    action_id: UUID
    lesson_id: int = Field(gt=0)
    skipped: bool
    occurred_at: datetime


class LessonQueueMoveIn(StrictModel):
    action_id: UUID
    direction: Literal["up", "down"]
    neighbor_lesson_id: int = Field(gt=0)


class LessonQueueMoveView(StrictModel):
    action_id: UUID
    lesson_id: int = Field(gt=0)
    direction: Literal["up", "down"]
    neighbor_lesson_id: int = Field(gt=0)
    occurred_at: datetime


class TextRequestIn(StrictModel):
    request_id: UUID
    topic: str = Field(min_length=1, max_length=200)

    @field_validator("topic")
    @classmethod
    def strip_topic(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return " ".join(value.split())


class TextRequestView(StrictModel):
    task_id: int = Field(gt=0)
    request_id: UUID
    topic: str
    state: GenerationTaskState
    created_at: datetime
    updated_at: datetime
    error: str | None
    lesson_id: int | None = Field(default=None, gt=0)


class TextPreparationView(StrictModel):
    task_id: int = Field(gt=0)
    state: Literal["pending", "running"]
    request_kind: GenerationRequestKind
    requested_topic: str | None = Field(default=None, min_length=1, max_length=200)
    waiting_for_capacity: bool
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_requested_topic(self) -> TextPreparationView:
        if self.request_kind == "topic_request" and self.requested_topic is None:
            raise ValueError("topic-request preparation requires requested_topic")
        if self.request_kind == "queue_fill" and self.requested_topic is not None:
            raise ValueError("queue-fill preparation must not include requested_topic")
        return self


class TextsState(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    texts: list[TextView]
    preparations: list[TextPreparationView]
    requests: list[TextRequestView]


class TextDetail(StrictModel):
    """A deliberately session-free lesson payload for no-evidence previewing."""

    lesson_id: int = Field(gt=0)
    lesson: LessonDocument
    term_bands: dict[str, TermKnowledgeBand]
    grammar_catalog: list[AgentGrammarCatalogEntry] = Field(default_factory=list)
    comfortable_grammar_construction_keys: list[str] = Field(default_factory=list)


class WordSurfaceForm(StrictModel):
    text: str = Field(min_length=1)
    occurrences: int = Field(ge=1)


class RelatedWordSense(StrictModel):
    """Another observed sense or grammatical role with the same written lemma."""

    key: str
    pos: str
    gloss: str
    pronunciation: str | None


class WordView(StrictModel):
    key: str
    lemma: str
    pos: str
    gloss: str
    pronunciation: str | None
    frequency_rank: int | None
    surface_forms: list[WordSurfaceForm] = Field(min_length=1)
    related_senses: list[RelatedWordSense] = Field(default_factory=list)
    occurrence_count: int = Field(ge=1)
    exposed_lesson_count: int = Field(ge=1)
    raw_reveal_count: int = Field(ge=0)
    counted_reveal_sessions: int = Field(ge=0)
    qualified_exposures: int = Field(ge=0)
    mastery: float = Field(ge=0.0, le=1.0)
    stability_days: float = Field(ge=0.0)
    reveal_failures: float = Field(ge=0)
    first_exposed_at: datetime
    last_exposed_at: datetime
    last_revealed_at: datetime | None
    next_due_at: datetime | None


class WordsState(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    words: list[WordView]


class CharacterView(StrictModel):
    """Observed use and replayable inferred recognition of one Han character."""

    character: str = Field(min_length=1, max_length=1)
    occurrence_count: int = Field(ge=1)
    exposed_lesson_count: int = Field(ge=1)
    raw_reveal_count: int = Field(ge=0)
    counted_reveal_sessions: int = Field(ge=0)
    distinct_word_contexts: int = Field(ge=0)
    qualified_exposures: int = Field(ge=0)
    inferred_failure_sessions: int = Field(ge=0)
    inferred_failure_mass: float = Field(ge=0.0)
    direct_successes: int = Field(ge=0)
    direct_failures: int = Field(ge=0)
    mastery: float = Field(ge=0.0, le=1.0)
    mastery_uncertainty: float = Field(ge=0.0, le=0.5)
    retrievability: float = Field(ge=0.0, le=1.0)
    stability_days: float = Field(ge=0.0)
    first_exposed_at: datetime
    last_exposed_at: datetime
    last_evidence_at: datetime | None
    next_due_at: datetime | None


class CharactersState(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    characters: list[CharacterView]


class StatisticsWordsSummary(StrictModel):
    total: int = Field(ge=0)
    learning: int = Field(ge=0)
    expected: int = Field(ge=0)
    familiar: int = Field(ge=0)
    mastered: int = Field(ge=0)
    expected_min_mastery: float = Field(ge=0.0, le=1.0)
    familiar_min_mastery: float = Field(ge=0.0, le=1.0)
    mastered_min_mastery: float = Field(ge=0.0, le=1.0)


class StatisticsReadingSummary(StrictModel):
    texts_read: int = Field(ge=0)
    completed_sessions: int = Field(ge=0)
    total_active_seconds: float = Field(ge=0.0)
    recent_average_wpm: float | None = Field(default=None, ge=0.0)
    recent_wpm_sessions: int = Field(ge=0)
    lifetime_average_wpm: float | None = Field(default=None, ge=0.0)
    lifetime_wpm_sessions: int = Field(ge=0)
    recent_window_size: int = Field(ge=1)
    minimum_wpm_active_seconds: float = Field(ge=0.0)
    minimum_wpm_completion_ratio: float = Field(ge=0.0, le=1.0)


class StatisticsLevelSummary(StrictModel):
    value: float = Field(ge=0.0, le=1.0)
    category: CefrLevel
    source: LevelSource
    status: CalibrationStatus
    lower: float | None = Field(default=None, ge=0.0, le=1.0)
    upper: float | None = Field(default=None, ge=0.0, le=1.0)
    lower_category: CefrLevel | None = None
    upper_category: CefrLevel | None = None
    qualified_attempts: int = Field(ge=0)
    usable_probes: int = Field(ge=0)
    qualified_readings: int = Field(default=0, ge=0)


class StatisticsSummary(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    words: StatisticsWordsSummary
    reading: StatisticsReadingSummary
    level: StatisticsLevelSummary


class GrammarExampleView(StrictModel):
    lesson_id: int = Field(gt=0)
    title: str
    sentence_key: str
    text: str
    translation: str | None = None
    note: str | None = None


class GrammarView(StrictModel):
    key: str
    label: str
    description: str
    category: str
    difficulty: float = Field(ge=0.0, le=1.0)
    occurrence_count: int = Field(ge=1)
    exposed_lesson_count: int = Field(ge=1)
    raw_help_count: int = Field(ge=0)
    counted_help_sessions: int = Field(ge=0)
    inferred_difficulty_signals: int = Field(ge=0)
    qualified_exposures: int = Field(ge=0)
    mastery: float = Field(ge=0.0, le=1.0)
    stability_days: float = Field(ge=0.0)
    first_exposed_at: datetime
    last_exposed_at: datetime
    last_helped_at: datetime | None
    next_due_at: datetime | None
    examples: list[GrammarExampleView] = Field(default_factory=list)


class GrammarState(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    constructions: list[GrammarView]
