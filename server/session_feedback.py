"""Reader feedback tags taken from rated sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast, get_args

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.models import (
    Interaction,
    Lesson,
)
from server.profile_state import ensure_profile
from server.schemas import (
    FeedbackTag,
)


def latest_profile_feedback(db: Session) -> list[FeedbackTag]:
    profile = ensure_profile(db)
    events = db.scalars(
        select(Interaction)
        .join(Lesson, Lesson.id == Interaction.lesson_id)
        .where(
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
            Interaction.event_type.in_(("lesson.completed", "lesson.rated")),
        )
        .order_by(Interaction.occurred_at, Interaction.id)
    ).all()
    return latest_completed_session_feedback(events)


def latest_completed_session_feedback(
    events: Sequence[Interaction],
) -> list[FeedbackTag]:
    latest_completion = next(
        (event for event in reversed(events) if event.event_type == "lesson.completed"),
        None,
    )
    if latest_completion is None:
        return []
    latest_rating = next(
        (
            event
            for event in reversed(events)
            if event.event_type == "lesson.rated"
            and event.lesson_id == latest_completion.lesson_id
            and event.session_id == latest_completion.session_id
        ),
        None,
    )
    return feedback_from_payload(latest_rating.payload) if latest_rating is not None else []


def feedback_from_payload(payload: Mapping[str, Any]) -> list[FeedbackTag]:
    value = payload.get("feedback")
    allowed = set(get_args(FeedbackTag))
    if not isinstance(value, list) or any(
        not isinstance(item, str) or item not in allowed for item in value
    ):
        return []
    return cast(list[FeedbackTag], list(value))
