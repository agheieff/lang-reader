"""Minimal persistence model for the local, single-profile reader."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from server.clock import as_utc, utc_now
from server.memory_model import MemoryState, knowledge
from server.schemas import CalibrationStatus, CefrLevel, GenerationTaskState


class Base(DeclarativeBase):
    pass


class Profile(Base):
    __tablename__ = "profile"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_profile_single_row"),
        CheckConstraint(
            "level IN ('A1', 'A2', 'B1', 'B2', 'C1', 'C2')",
            name="ck_profile_cefr_level",
        ),
        CheckConstraint(
            "difficulty >= 0.0 AND difficulty <= 1.0",
            name="ck_profile_difficulty",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    learning_language: Mapped[str] = mapped_column(String(35), default="es-ES")
    translation_language: Mapped[str] = mapped_column(String(35), default="en")
    level: Mapped[CefrLevel] = mapped_column(String(2), default="A1")
    difficulty: Mapped[float] = mapped_column(Float, default=0.15)
    interests: Mapped[list[str]] = mapped_column(JSON, default=list)
    preferences: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class Lesson(Base):
    """Imported lesson with indexed columns materialized from the canonical JSON payload."""

    __tablename__ = "lessons"
    __table_args__ = (
        Index("ix_lessons_languages", "learning_language", "translation_language"),
        CheckConstraint(
            "level IN ('A1', 'A2', 'B1', 'B2', 'C1', 'C2')",
            name="ck_lessons_cefr_level",
        ),
        CheckConstraint(
            "difficulty >= 0.0 AND difficulty <= 1.0",
            name="ck_lessons_difficulty",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    schema_version: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(300))
    learning_language: Mapped[str] = mapped_column(String(35), index=True)
    translation_language: Mapped[str] = mapped_column(String(35))
    topic: Mapped[str | None] = mapped_column(String(200), nullable=True)
    level: Mapped[CefrLevel] = mapped_column(String(2), default="A1")
    difficulty: Mapped[float] = mapped_column(Float, default=0.15)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64))
    source_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    # Predicted fraction of known running words when imported; drives the difficulty loop.
    known_share_at_import: Mapped[float | None] = mapped_column(Float, nullable=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('lesson.started', 'term.revealed', "
            "'translation.revealed', 'lesson.completed', 'lesson.rated')",
            name="ck_interactions_event_type",
        ),
        Index("ix_interactions_lesson_session", "lesson_id", "session_id", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(100), index=True)
    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("lessons.id", ondelete="RESTRICT"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(40))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class LessonQueueAction(Base):
    """Append-only, non-learning changes to whether a lesson is in the reading queue."""

    __tablename__ = "lesson_queue_actions"
    __table_args__ = (Index("ix_lesson_queue_actions_lesson_id", "lesson_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id", ondelete="RESTRICT"))
    skipped: Mapped[bool] = mapped_column(Boolean)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class LessonQueueMoveAction(Base):
    """Append-only swaps in the latent lesson order."""

    __tablename__ = "lesson_queue_move_actions"
    __table_args__ = (
        CheckConstraint("direction IN ('up', 'down')", name="ck_lesson_queue_move_direction"),
        Index("ix_lesson_queue_move_lesson_id", "lesson_id", "id"),
        Index("ix_lesson_queue_move_neighbor_id", "neighbor_lesson_id", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id", ondelete="RESTRICT"))
    neighbor_lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id", ondelete="RESTRICT"))
    direction: Mapped[str] = mapped_column(String(4))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class GenerationTask(Base):
    __tablename__ = "generation_tasks"
    __table_args__ = (
        CheckConstraint(
            "state IN ('pending', 'running', 'completed', 'failed')",
            name="ck_generation_tasks_state",
        ),
        Index("ix_generation_tasks_state_created", "state", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    state: Mapped[GenerationTaskState] = mapped_column(String(20), default="pending")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    dedupe_key: Mapped[str | None] = mapped_column(
        String(100), unique=True, nullable=True, default=None
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    log_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AudioTask(Base):
    __tablename__ = "audio_tasks"
    __table_args__ = (
        CheckConstraint(
            "state IN ('pending', 'running', 'completed', 'failed', 'superseded')",
            name="ck_audio_tasks_state",
        ),
        UniqueConstraint("lesson_id", "cache_key", name="uq_audio_tasks_lesson_cache"),
        Index("ix_audio_tasks_state_created", "state", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("lessons.id", ondelete="RESTRICT"), index=True
    )
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    provider: Mapped[str] = mapped_column(String(40))
    model_revision: Mapped[str] = mapped_column(String(100))
    learning_language: Mapped[str] = mapped_column(String(35))
    voice_id: Mapped[str] = mapped_column(String(40))
    state: Mapped[str] = mapped_column(String(20), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    relative_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ProficiencyState(Base):
    __tablename__ = "proficiency_state"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_proficiency_state_single_row"),
        CheckConstraint(
            "status IN ('unstarted', 'collecting', 'rough', 'stable')",
            name="ck_proficiency_state_status",
        ),
        CheckConstraint(
            "estimate IS NULL OR (estimate >= 0.0 AND estimate <= 1.0)",
            name="ck_proficiency_state_estimate",
        ),
        CheckConstraint(
            "lower IS NULL OR (lower >= 0.0 AND lower <= 1.0)",
            name="ck_proficiency_state_lower",
        ),
        CheckConstraint(
            "upper IS NULL OR (upper >= 0.0 AND upper <= 1.0)",
            name="ck_proficiency_state_upper",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    status: Mapped[CalibrationStatus] = mapped_column(String(20), default="unstarted")
    estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    level: Mapped[CefrLevel | None] = mapped_column(String(2), nullable=True)
    lower_level: Mapped[CefrLevel | None] = mapped_column(String(2), nullable=True)
    upper_level: Mapped[CefrLevel | None] = mapped_column(String(2), nullable=True)
    qualified_attempts: Mapped[int] = mapped_column(Integer, default=0)
    usable_probes: Mapped[int] = mapped_column(Integer, default=0)
    qualified_readings: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class LexemeState(Base):
    __tablename__ = "lexeme_states"

    learning_language: Mapped[str] = mapped_column(String(35), primary_key=True)
    translation_language: Mapped[str] = mapped_column(String(35), primary_key=True)
    term_key: Mapped[str] = mapped_column(String(200), primary_key=True)
    lemma: Mapped[str] = mapped_column(String(200), index=True)
    pos: Mapped[str] = mapped_column(String(40))
    gloss: Mapped[str] = mapped_column(String(500))
    pronunciation: Mapped[str | None] = mapped_column(String(300), nullable=True)
    frequency_rank: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    # alpha/beta tally weighted success/failure evidence (uncertainty only); memory_difficulty,
    # stability_days and last_seen_at are the FSRS state, and prior_known covers no evidence.
    alpha: Mapped[float] = mapped_column(Float, default=2.0)
    beta: Mapped[float] = mapped_column(Float, default=2.0)
    stability_days: Mapped[float] = mapped_column(Float, default=0.5)
    memory_difficulty: Mapped[float | None] = mapped_column(Float, nullable=True)
    prior_known: Mapped[float] = mapped_column(Float, default=0.5)
    qualified_exposures: Mapped[int] = mapped_column(Integer, default=0)
    reveal_failures: Mapped[float] = mapped_column(Float, default=0.0)
    distinct_lessons: Mapped[int] = mapped_column(Integer, default=0)

    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_revealed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    next_due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    @property
    def memory_state(self) -> MemoryState | None:
        if self.memory_difficulty is None or self.last_seen_at is None:
            return None
        return MemoryState(self.memory_difficulty, self.stability_days, as_utc(self.last_seen_at))

    @property
    def mastery(self) -> float:
        """Probability of still recalling this word a month from now (time-aware)."""

        return knowledge(self.memory_state, prior=self.prior_known, at=utc_now())


class CharacterState(Base):
    """Replayable inferred recognition state for one literal Han character."""

    __tablename__ = "character_states"
    __table_args__ = (Index("ix_character_states_due", "learning_language", "next_due_at"),)

    learning_language: Mapped[str] = mapped_column(String(35), primary_key=True)
    translation_language: Mapped[str] = mapped_column(String(35), primary_key=True)
    character: Mapped[str] = mapped_column(String(8), primary_key=True)

    alpha: Mapped[float] = mapped_column(Float, default=2.0)
    beta: Mapped[float] = mapped_column(Float, default=2.0)
    stability_days: Mapped[float] = mapped_column(Float, default=0.5)
    memory_difficulty: Mapped[float | None] = mapped_column(Float, nullable=True)
    qualified_exposures: Mapped[int] = mapped_column(Integer, default=0)
    inferred_failure_sessions: Mapped[int] = mapped_column(Integer, default=0)
    inferred_failure_mass: Mapped[float] = mapped_column(Float, default=0.0)
    direct_successes: Mapped[int] = mapped_column(Integer, default=0)
    direct_failures: Mapped[int] = mapped_column(Integer, default=0)
    distinct_lessons: Mapped[int] = mapped_column(Integer, default=0)
    distinct_word_contexts: Mapped[int] = mapped_column(Integer, default=0)

    first_evidence_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_evidence_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_inferred_failure_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    next_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    @property
    def memory_state(self) -> MemoryState | None:
        if self.memory_difficulty is None or self.last_evidence_at is None:
            return None
        return MemoryState(
            self.memory_difficulty, self.stability_days, as_utc(self.last_evidence_at)
        )

    @property
    def mastery(self) -> float:
        """Probability of still recognizing this character a month from now (time-aware)."""

        return knowledge(self.memory_state, prior=0.5, at=utc_now())


class GrammarState(Base):
    """Replayable learning state for one authored grammar construction."""

    __tablename__ = "grammar_states"
    __table_args__ = (
        CheckConstraint(
            "difficulty >= 0.0 AND difficulty <= 1.0",
            name="ck_grammar_states_difficulty",
        ),
        Index("ix_grammar_states_due", "learning_language", "next_due_at"),
    )

    learning_language: Mapped[str] = mapped_column(String(35), primary_key=True)
    translation_language: Mapped[str] = mapped_column(String(35), primary_key=True)
    construction_key: Mapped[str] = mapped_column(String(200), primary_key=True)
    label: Mapped[str] = mapped_column(String(300), index=True)
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(100), index=True)
    difficulty: Mapped[float] = mapped_column(Float, index=True)

    alpha: Mapped[float] = mapped_column(Float, default=2.0)
    beta: Mapped[float] = mapped_column(Float, default=2.0)
    stability_days: Mapped[float] = mapped_column(Float, default=0.5)
    qualified_exposures: Mapped[int] = mapped_column(Integer, default=0)
    explicit_help_failures: Mapped[int] = mapped_column(Integer, default=0)
    inferred_difficulty_signals: Mapped[int] = mapped_column(Integer, default=0)
    distinct_lessons: Mapped[int] = mapped_column(Integer, default=0)

    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_helped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    @property
    def mastery(self) -> float:
        total = self.alpha + self.beta
        return self.alpha / total if total else 0.5


class DerivedState(Base):
    """Evidence watermark of the last complete derived-cache rebuild."""

    __tablename__ = "derived_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    watermark: Mapped[str] = mapped_column(String(64))
    rebuilt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ReadingPreferenceRevision(Base):
    """Append-only reading-preference notes; the latest revision is current."""

    __tablename__ = "reading_preference_revisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(10))  # "user" or "agent"
    reason: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PreferenceMessage(Base):
    """A message from the learner that the agent folds into the preference notes."""

    __tablename__ = "preference_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    handled_revision_id: Mapped[int | None] = mapped_column(
        ForeignKey("reading_preference_revisions.id"), nullable=True
    )


class DomainRejection(Base):
    """A prose draft the host refused for its subject domain; the task's own row shows the rest."""

    __tablename__ = "domain_rejections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(Integer, index=True)
    attempt: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(10))  # "missing", "repeat" or "duplicate"
    declared: Mapped[list[str | None]] = mapped_column(JSON)
    avoid_domains: Mapped[list[str]] = mapped_column(JSON)
    allowed_domains: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
