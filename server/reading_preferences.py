"""Reading-preference notes, learner messages about them, and the content history they draw on.

The notes are one free-text document, edited by the learner and mostly maintained by the agent.
Revisions are append-only, so every change stays inspectable. Learner messages never edit the
notes directly: the agent folds them in on its next update. Updates are based on one revision; an
update computed from a revision that has since been superseded is discarded and recomputed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast, get_args

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.clock import as_utc
from server.lesson_activity import lesson_activities, profile_lessons
from server.lesson_content import lesson_document
from server.models import Interaction, PreferenceMessage, Profile, ReadingPreferenceRevision
from server.schemas import (
    ContentHistoryItem,
    ContentMove,
    ContentReaction,
    FeedbackTag,
    PendingPreferenceMessage,
    ReadingPreferencesView,
    SubjectDomain,
)

CONTENT_HISTORY_LIMIT = 20
# Reactions (finished, rated, skipped texts) since the last agent revision that trigger an update.
REACTIONS_PER_UPDATE = 3
_FEEDBACK_TAGS = set(get_args(FeedbackTag))


class PreferencesConflictError(ValueError):
    pass


def current_revision(db: Session) -> ReadingPreferenceRevision | None:
    return db.scalar(
        select(ReadingPreferenceRevision).order_by(ReadingPreferenceRevision.id.desc()).limit(1)
    )


def default_preferences(db: Session) -> str:
    profile = db.get(Profile, 1)
    interests = profile.interests if profile is not None else []
    listed = ", ".join(interests) if interests else "none listed yet"
    return f"Interests: {listed}.\n"


def current_preferences(db: Session) -> str:
    revision = current_revision(db)
    return revision.text if revision is not None else default_preferences(db)


def preferences_view(db: Session) -> ReadingPreferencesView:
    revision = current_revision(db)
    last_agent = db.scalar(
        select(ReadingPreferenceRevision)
        .where(ReadingPreferenceRevision.source == "agent")
        .order_by(ReadingPreferenceRevision.id.desc())
        .limit(1)
    )
    return ReadingPreferencesView(
        text=revision.text if revision is not None else default_preferences(db),
        revision_id=revision.id if revision is not None else None,
        source=cast(Any, revision.source) if revision is not None else "default",
        updated_at=as_utc(revision.created_at) if revision is not None else None,
        last_agent_reason=last_agent.reason if last_agent is not None else None,
        pending_messages=[
            PendingPreferenceMessage(text=message.text, created_at=as_utc(message.created_at))
            for message in pending_messages(db)
        ],
    )


def save_user_preferences(
    db: Session, text: str, *, expected_revision_id: int | None
) -> ReadingPreferencesView:
    revision = current_revision(db)
    if (revision.id if revision is not None else None) != expected_revision_id:
        raise PreferencesConflictError("reading preferences changed; reload before saving")
    db.add(ReadingPreferenceRevision(text=text.strip() + "\n", source="user"))
    db.commit()
    return preferences_view(db)


def add_preference_message(db: Session, message_id: str, text: str) -> ReadingPreferencesView:
    existing = db.scalar(
        select(PreferenceMessage).where(PreferenceMessage.message_id == message_id)
    )
    if existing is None:
        db.add(PreferenceMessage(message_id=message_id, text=text))
        db.commit()
    elif existing.text != text:
        raise PreferencesConflictError("message_id reused with different text")
    return preferences_view(db)


def pending_messages(db: Session) -> list[PreferenceMessage]:
    return list(
        db.scalars(
            select(PreferenceMessage)
            .where(PreferenceMessage.handled_revision_id.is_(None))
            .order_by(PreferenceMessage.id)
        )
    )


def content_history(db: Session, limit: int = CONTENT_HISTORY_LIMIT) -> list[ContentHistoryItem]:
    """The most recent texts, newest first, with their content move and the learner's reaction."""

    profile = db.get(Profile, 1)
    if profile is None:
        return []
    lessons = profile_lessons(db, profile)
    recent = sorted(lessons, key=lambda lesson: (as_utc(lesson.imported_at), lesson.id))
    recent = list(reversed(recent))[:limit]
    activity = lesson_activities(db, [lesson.id for lesson in recent])
    feedback = _latest_feedback(db, [lesson.id for lesson in recent])
    items: list[ContentHistoryItem] = []
    for lesson in recent:
        document = lesson_document(lesson)
        if document.calibration is not None:
            continue
        metadata = document.metadata
        plan = metadata.get("content_plan")
        move = plan.get("move") if isinstance(plan, dict) else None
        domain = metadata.get("content_domain")
        if metadata.get("requested_topic"):
            move = "requested"
        state = activity[lesson.id]
        items.append(
            ContentHistoryItem(
                title=document.title,
                topic=document.topic,
                move=cast(ContentMove, move) if move in _MOVES else None,
                angle=_text(metadata.get("content_angle")),
                domain=cast(SubjectDomain, domain) if domain in _DOMAINS else None,
                hypothesis=_text(metadata.get("content_hypothesis")),
                reaction=_reaction(state.rating, state.completed, state.skipped, state.opened_at),
                feedback=feedback.get(lesson.id, []),
                rereads=max(0, len(state.completion_session_ids) - 1),
            )
        )
    return items


def update_due(db: Session) -> bool:
    """Pending learner messages, or enough new reactions since the agent last updated."""

    if pending_messages(db):
        return True
    since = db.scalar(
        select(ReadingPreferenceRevision.created_at)
        .where(ReadingPreferenceRevision.source == "agent")
        .order_by(ReadingPreferenceRevision.id.desc())
        .limit(1)
    )
    return _reactions_since(db, as_utc(since) if since is not None else None) >= (
        REACTIONS_PER_UPDATE
    )


def apply_agent_update(
    db: Session,
    *,
    base_revision_id: int | None,
    text: str,
    reason: str,
    message_ids: Sequence[int],
) -> bool:
    """Store the agent's revision unless the notes changed since it read them."""

    revision = current_revision(db)
    if (revision.id if revision is not None else None) != base_revision_id:
        return False
    stored = ReadingPreferenceRevision(text=text.strip() + "\n", source="agent", reason=reason)
    db.add(stored)
    db.flush()
    for message in db.scalars(
        select(PreferenceMessage).where(PreferenceMessage.id.in_(message_ids))
    ):
        message.handled_revision_id = stored.id
    db.commit()
    return True


_MOVES = {"favourite", "variation", "new", "requested"}
_DOMAINS = set(get_args(SubjectDomain))


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _reaction(
    rating: int | None, completed: bool, skipped: bool, opened_at: datetime | None
) -> ContentReaction:
    if rating == 1:
        return "liked"
    if rating == -1:
        return "disliked"
    if skipped:
        return "skipped"
    if completed:
        return "finished"
    return "abandoned" if opened_at is not None else "unread"


def _latest_feedback(db: Session, lesson_ids: list[int]) -> dict[int, list[FeedbackTag]]:
    result: dict[int, list[FeedbackTag]] = {}
    if not lesson_ids:
        return result
    for event in db.scalars(
        select(Interaction)
        .where(Interaction.lesson_id.in_(lesson_ids), Interaction.event_type == "lesson.rated")
        .order_by(Interaction.occurred_at, Interaction.id)
    ):
        tags = event.payload.get("feedback")
        if isinstance(tags, list):
            result[event.lesson_id] = [
                cast(FeedbackTag, tag) for tag in tags if tag in _FEEDBACK_TAGS
            ]
    return result


def _reactions_since(db: Session, since: datetime | None) -> int:
    query = select(Interaction.lesson_id).where(
        Interaction.event_type.in_(("lesson.completed", "lesson.rated"))
    )
    if since is not None:
        query = query.where(Interaction.recorded_at > since)
    lessons = set(db.scalars(query))
    return len(lessons) + _skips_since(db, since)


def _skips_since(db: Session, since: datetime | None) -> int:
    from server.models import LessonQueueAction

    query = select(LessonQueueAction.lesson_id).where(LessonQueueAction.skipped.is_(True))
    if since is not None:
        query = query.where(LessonQueueAction.recorded_at > since)
    return len(set(db.scalars(query)))
