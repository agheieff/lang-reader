"""What generation is told about the learner: briefs, vocabulary plan, preferences."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    Field,
    field_validator,
)

from server.schemas.base import (
    CefrLevel,
    FeedbackTag,
    StrictModel,
    SubjectDomain,
)
from server.schemas.views import (
    AgentGrammarCatalogEntry,
    ProfileView,
)


class AgentTermBrief(StrictModel):
    key: str
    lemma: str
    pos: str
    gloss: str
    pronunciation: str | None
    frequency_rank: int | None
    mastery: float
    stability_days: float
    qualified_exposures: int
    reveal_failures: float
    next_due_at: datetime | None
    urgency: float
    reason: Literal["unseen", "due", "fragile"]


class AgentGrammarBrief(StrictModel):
    key: str
    label: str
    description: str
    category: str
    difficulty: float = Field(ge=0.0, le=1.0)
    mastery: float = Field(ge=0.0, le=1.0)
    stability_days: float = Field(ge=0.0)
    qualified_exposures: int = Field(ge=0)
    help_failures: int = Field(ge=0)
    inferred_difficulty_signals: int = Field(ge=0)
    next_due_at: datetime | None
    urgency: float
    reason: Literal["unseen", "due", "fragile"]


class AgentLessonBrief(StrictModel):
    id: int
    key: str
    title: str
    topic: str | None
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    imported_at: datetime
    completed: bool
    rating: Literal[-1, 1] | None
    feedback: list[FeedbackTag]
    target_term_keys: list[str]
    grammar_keys: list[str] = Field(default_factory=list)
    opening_excerpt: str = Field(max_length=180)
    ending_excerpt: str = Field(max_length=180)
    calibration_lesson: bool = False
    metadata: dict[str, Any]
    known_share: float | None = Field(default=None, ge=0.0, le=1.0)


class NewWordCandidate(StrictModel):
    lemma: str = Field(min_length=1, max_length=200)
    pronunciation: str | None = Field(default=None, max_length=300)
    gloss: str = Field(max_length=500)
    frequency_rank: int | None = Field(default=None, ge=1)
    level: int | None = Field(default=None, ge=1)
    source: str = Field(min_length=1, max_length=40)


class VocabularyPlan(StrictModel):
    """New-vocabulary guidance: list words near the frontier plus the agent's own picks."""

    list_candidates: list[NewWordCandidate] = Field(default_factory=list, max_length=32)
    list_new_words: int = Field(ge=0, le=12)
    free_new_words: int = Field(ge=0, le=12)
    list_exhausted: bool = False
    target_known_share: float = Field(ge=0.5, le=1.0)
    recent_known_share: float | None = Field(default=None, ge=0.0, le=1.0)


ContentMove = Literal["favourite", "variation", "new", "requested"]
ContentReaction = Literal["liked", "disliked", "finished", "abandoned", "skipped", "unread"]


class ContentHistoryItem(StrictModel):
    """One recent text's subject and how the learner reacted to it."""

    title: str
    topic: str | None = None
    move: ContentMove | None = None
    angle: str | None = None
    domain: SubjectDomain | None = None
    hypothesis: str | None = None
    reaction: ContentReaction
    feedback: list[FeedbackTag] = Field(default_factory=list)
    rereads: int = Field(default=0, ge=0)


class PendingPreferenceMessage(StrictModel):
    text: str
    created_at: datetime


class ReadingPreferencesView(StrictModel):
    text: str
    revision_id: int | None = None
    source: Literal["default", "user", "agent"]
    updated_at: datetime | None = None
    last_agent_reason: str | None = None
    pending_messages: list[PendingPreferenceMessage] = Field(default_factory=list)


class ReadingPreferencesUpdate(StrictModel):
    text: str = Field(max_length=4_000)
    expected_revision_id: int | None = None


class PreferenceMessageIn(StrictModel):
    message_id: UUID
    text: str = Field(min_length=1, max_length=1_000)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class AgentBrief(StrictModel):
    schema_version: Literal[1] = 1
    generated_at: datetime
    profile: ProfileView
    lesson_count: int
    completed_lesson_count: int
    interaction_count: int
    language_guidance: list[str]
    grammar_catalog: list[AgentGrammarCatalogEntry] = Field(default_factory=list)
    priority_grammar: list[AgentGrammarBrief] = Field(default_factory=list)
    priority_terms: list[AgentTermBrief]
    mastered_term_keys: list[str]
    recent_lessons: list[AgentLessonBrief]
    vocabulary: VocabularyPlan | None = None
    reading_preferences: str = ""
    content_history: list[ContentHistoryItem] = Field(default_factory=list)
