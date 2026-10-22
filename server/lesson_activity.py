"""Shared read-side projection over append-only lesson interactions and queue actions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.clock import as_utc, earliest_datetime, latest_datetime
from server.models import Interaction, Lesson, LessonQueueAction, Profile
from server.schemas import TextStatus


def require_profile(db: Session) -> Profile:
    profile = db.get(Profile, 1)
    if profile is None:
        raise LookupError("profile is not configured")
    return profile


def profile_lessons(db: Session, profile: Profile) -> list[Lesson]:
    """Return the profile's lessons in durable import order."""

    return list(
        db.scalars(
            select(Lesson)
            .where(
                Lesson.learning_language == profile.learning_language,
                Lesson.translation_language == profile.translation_language,
            )
            .order_by(Lesson.imported_at, Lesson.id)
        ).all()
    )


def opened_lesson_interactions(
    db: Session, lessons: Sequence[Lesson]
) -> tuple[list[Lesson], list[Interaction]]:
    """Return lessons proven opened by any stored interaction, plus their ordered events."""

    lesson_ids = [lesson.id for lesson in lessons]
    if not lesson_ids:
        return [], []
    opened_ids = set(
        db.scalars(
            select(Interaction.lesson_id).where(Interaction.lesson_id.in_(lesson_ids)).distinct()
        ).all()
    )
    opened = [lesson for lesson in lessons if lesson.id in opened_ids]
    if not opened:
        return [], []
    interactions = db.scalars(
        select(Interaction)
        .where(Interaction.lesson_id.in_([lesson.id for lesson in opened]))
        .order_by(Interaction.occurred_at, Interaction.id)
    ).all()
    return opened, list(interactions)


@dataclass(frozen=True)
class LessonDisposition:
    """Minimal completion/queue state used by reader and generation selection."""

    lesson_id: int
    completed: bool = False
    latest_interaction_recorded_at: datetime | None = None
    latest_queue_action: LessonQueueAction | None = None

    @property
    def skipped(self) -> bool:
        return self.latest_queue_action is not None and self.latest_queue_action.skipped

    @property
    def ready(self) -> bool:
        if self.completed or self.skipped:
            return False
        if self.latest_interaction_recorded_at is None:
            return True
        action = self.latest_queue_action
        return (
            action is not None and as_utc(action.recorded_at) > self.latest_interaction_recorded_at
        )

    @property
    def unread(self) -> bool:
        return not self.completed and not self.skipped


@dataclass
class LessonActivity:
    """Everything needed to classify one lesson without interpreting learning evidence."""

    lesson_id: int
    session_ids: set[str] = field(default_factory=set)
    completion_session_ids: set[str] = field(default_factory=set)
    opened_at: datetime | None = None
    last_completed_at: datetime | None = None
    rating: Literal[-1, 1] | None = None
    latest_interaction_id: int | None = None
    latest_interaction_recorded_at: datetime | None = None
    latest_queue_action: LessonQueueAction | None = None

    @property
    def completed(self) -> bool:
        return bool(self.completion_session_ids)

    @property
    def skipped(self) -> bool:
        return self.disposition.skipped

    @property
    def skipped_at(self) -> datetime | None:
        action = self.latest_queue_action
        return as_utc(action.occurred_at) if action is not None and action.skipped else None

    @property
    def ready(self) -> bool:
        """Whether this lesson is in the current untouched/restore queue epoch."""

        return self.disposition.ready

    @property
    def disposition(self) -> LessonDisposition:
        return LessonDisposition(
            lesson_id=self.lesson_id,
            completed=self.completed,
            latest_interaction_recorded_at=self.latest_interaction_recorded_at,
            latest_queue_action=self.latest_queue_action,
        )

    @property
    def unread(self) -> bool:
        """Whether the default reader may open or resume this lesson."""

        return self.disposition.unread

    @property
    def status(self) -> TextStatus:
        if self.completed:
            return "read"
        if self.skipped:
            return "skipped"
        if self.ready:
            return "queued"
        if self.session_ids:
            return "in_progress"
        return "queued"


def lesson_activities(
    db: Session,
    lesson_ids: Collection[int],
) -> dict[int, LessonActivity]:
    """Project one consistent activity/status snapshot for the requested lessons."""

    selected = tuple(dict.fromkeys(lesson_ids))
    activities = {lesson_id: LessonActivity(lesson_id) for lesson_id in selected}
    if not selected:
        return activities

    interactions = db.scalars(
        select(Interaction)
        .where(Interaction.lesson_id.in_(selected))
        .order_by(Interaction.occurred_at, Interaction.id)
    ).all()
    for interaction in interactions:
        activity = activities[interaction.lesson_id]
        occurred_at = as_utc(interaction.occurred_at)
        activity.session_ids.add(interaction.session_id)
        activity.opened_at = earliest_datetime(activity.opened_at, occurred_at)
        if interaction.event_type == "lesson.completed":
            activity.completion_session_ids.add(interaction.session_id)
            activity.last_completed_at = latest_datetime(
                activity.last_completed_at,
                occurred_at,
            )
        elif interaction.event_type == "lesson.rated":
            rating = interaction.payload.get("rating")
            if rating in {-1, 1}:
                activity.rating = rating
        if (
            activity.latest_interaction_id is None
            or interaction.id > activity.latest_interaction_id
        ):
            activity.latest_interaction_id = interaction.id
            activity.latest_interaction_recorded_at = as_utc(interaction.recorded_at)

    for lesson_id, action in latest_lesson_queue_actions(db, selected).items():
        activities[lesson_id].latest_queue_action = action
    return activities


def latest_lesson_queue_actions(
    db: Session,
    lesson_ids: Collection[int] | None = None,
) -> dict[int, LessonQueueAction]:
    """Return the latest server-accepted queue action for each requested lesson."""

    statement = select(LessonQueueAction).order_by(LessonQueueAction.id)
    if lesson_ids is not None:
        selected = tuple(dict.fromkeys(lesson_ids))
        if not selected:
            return {}
        statement = statement.where(LessonQueueAction.lesson_id.in_(selected))
    latest: dict[int, LessonQueueAction] = {}
    for action in db.scalars(statement).all():
        latest[action.lesson_id] = action
    return latest


def lesson_dispositions(
    db: Session,
    lesson_ids: Collection[int],
) -> dict[int, LessonDisposition]:
    """Project queue eligibility without loading interaction payloads or session history."""

    selected = tuple(dict.fromkeys(lesson_ids))
    if not selected:
        return {}
    latest_interactions: dict[int, datetime] = {
        lesson_id: recorded_at
        for lesson_id, recorded_at in db.execute(
            select(
                Interaction.lesson_id,
                func.max(Interaction.recorded_at),
            )
            .where(Interaction.lesson_id.in_(selected))
            .group_by(Interaction.lesson_id)
        )
        .tuples()
        .all()
        if recorded_at is not None
    }
    completed = set(
        db.scalars(
            select(Interaction.lesson_id)
            .where(
                Interaction.lesson_id.in_(selected),
                Interaction.event_type == "lesson.completed",
            )
            .distinct()
        ).all()
    )
    actions = latest_lesson_queue_actions(db, selected)
    return {
        lesson_id: LessonDisposition(
            lesson_id=lesson_id,
            completed=lesson_id in completed,
            latest_interaction_recorded_at=(
                as_utc(latest_interactions[lesson_id]) if lesson_id in latest_interactions else None
            ),
            latest_queue_action=actions.get(lesson_id),
        )
        for lesson_id in selected
    }


def lesson_exposure_times(
    interactions: Sequence[Interaction],
) -> dict[int, tuple[datetime, datetime]]:
    """Return first/last session-opening evidence per lesson."""

    session_opened_at: dict[tuple[int, str], datetime] = {}
    for interaction in interactions:
        session = (interaction.lesson_id, interaction.session_id)
        session_opened_at[session] = earliest_datetime(
            session_opened_at.get(session),
            as_utc(interaction.occurred_at),
        )

    per_lesson: dict[int, list[datetime]] = defaultdict(list)
    for (lesson_id, _session_id), opened_at in session_opened_at.items():
        per_lesson[lesson_id].append(opened_at)
    return {
        lesson_id: (min(opened_at), max(opened_at)) for lesson_id, opened_at in per_lesson.items()
    }
