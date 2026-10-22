"""Append-only queue disposition and ordering, separate from learning evidence."""

from __future__ import annotations

import fcntl
import threading
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from typing import Literal, cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.clock import as_utc, utc_now
from server.lesson_activity import (
    latest_lesson_queue_actions,
    lesson_dispositions,
)
from server.models import (
    Interaction,
    Lesson,
    LessonQueueAction,
    LessonQueueMoveAction,
    Profile,
)
from server.schemas import LessonQueueActionView, LessonQueueMoveView
from server.workspaces import Workspace


class LessonQueueConflictError(ValueError):
    pass


QueueMoveDirection = Literal["up", "down"]
_fallback_queue_move_lock = threading.Lock()


@contextmanager
def _serialized_queue_moves(db: Session) -> Iterator[None]:
    """Serialize the validate-and-append step across app processes for one profile."""

    workspace = db.info.get("workspace")
    if not isinstance(workspace, Workspace):
        with _fallback_queue_move_lock:
            yield
        return

    lock_path = workspace.directory / "queue-moves.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path.parent.chmod(0o700)
    with lock_path.open("a", encoding="utf-8") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def skipped_lesson_ids(db: Session, lesson_ids: Collection[int] | None = None) -> set[int]:
    return {
        lesson_id
        for lesson_id, action in latest_lesson_queue_actions(db, lesson_ids).items()
        if action.skipped
    }


def ordered_profile_lessons(db: Session) -> list[Lesson]:
    """Replay durable adjacent swaps over the profile's complete import order."""

    profile = db.get(Profile, 1)
    if profile is None:
        raise LookupError("profile is not configured")
    lessons = list(
        db.scalars(
            select(Lesson)
            .where(
                Lesson.learning_language == profile.learning_language,
                Lesson.translation_language == profile.translation_language,
            )
            .order_by(Lesson.imported_at, Lesson.id)
        ).all()
    )
    lesson_by_id = {lesson.id: lesson for lesson in lessons}
    order = list(lesson_by_id)
    positions = {lesson_id: index for index, lesson_id in enumerate(order)}
    for action in db.scalars(
        select(LessonQueueMoveAction).order_by(LessonQueueMoveAction.id)
    ).all():
        if action.lesson_id not in lesson_by_id or action.neighbor_lesson_id not in lesson_by_id:
            continue
        lesson_index = positions[action.lesson_id]
        neighbor_index = positions[action.neighbor_lesson_id]
        order[lesson_index], order[neighbor_index] = order[neighbor_index], order[lesson_index]
        positions[action.lesson_id] = neighbor_index
        positions[action.neighbor_lesson_id] = lesson_index
    return [lesson_by_id[lesson_id] for lesson_id in order]


def ready_lesson_ids(db: Session) -> list[int]:
    return [lesson.id for lesson in ready_lessons(db)]


def ready_lessons(db: Session) -> list[Lesson]:
    """Return lessons in the current Ready epoch, in replayed queue order.

    A restore starts a new queue epoch without deleting earlier learning evidence. The lesson
    remains Ready until the browser records another interaction after that restore.
    """

    ordered = ordered_profile_lessons(db)
    if not ordered:
        return []
    dispositions = lesson_dispositions(db, [lesson.id for lesson in ordered])
    return [lesson for lesson in ordered if dispositions[lesson.id].ready]


def unread_lessons(db: Session) -> list[Lesson]:
    """Return resumable and untouched lessons, excluding completed and currently skipped texts."""

    ordered = ordered_profile_lessons(db)
    dispositions = lesson_dispositions(db, [lesson.id for lesson in ordered])
    return [lesson for lesson in ordered if dispositions[lesson.id].unread]


def record_lesson_queue_action(
    db: Session,
    *,
    lesson_id: int,
    action_id: UUID,
    skipped: bool,
) -> LessonQueueActionView:
    """Append one idempotent skip/restore command without creating learning evidence."""

    lesson = _profile_lesson(db, lesson_id)
    stored_action_id = str(action_id)
    existing = db.scalar(
        select(LessonQueueAction).where(LessonQueueAction.action_id == stored_action_id)
    )
    if existing is not None:
        return _matching_action(existing, lesson_id=lesson.id, skipped=skipped)

    if (
        db.scalar(
            select(Interaction.id)
            .where(
                Interaction.lesson_id == lesson.id,
                Interaction.event_type == "lesson.completed",
            )
            .limit(1)
        )
        is not None
    ):
        action_name = "skip" if skipped else "restore"
        raise LessonQueueConflictError(f"cannot {action_name} a completed lesson")

    action = LessonQueueAction(
        action_id=stored_action_id,
        lesson_id=lesson.id,
        skipped=skipped,
        occurred_at=utc_now(),
    )
    db.add(action)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        concurrent = db.scalar(
            select(LessonQueueAction).where(LessonQueueAction.action_id == stored_action_id)
        )
        if concurrent is None:
            raise
        return _matching_action(concurrent, lesson_id=lesson.id, skipped=skipped)
    db.refresh(action)
    return _action_view(action)


def record_lesson_queue_move(
    db: Session,
    *,
    lesson_id: int,
    action_id: UUID,
    direction: QueueMoveDirection,
    neighbor_lesson_id: int,
) -> LessonQueueMoveView:
    """Append one idempotent adjacent swap between two currently Ready lessons."""

    with _serialized_queue_moves(db):
        return _record_lesson_queue_move(
            db,
            lesson_id=lesson_id,
            action_id=action_id,
            direction=direction,
            neighbor_lesson_id=neighbor_lesson_id,
        )


def _record_lesson_queue_move(
    db: Session,
    *,
    lesson_id: int,
    action_id: UUID,
    direction: QueueMoveDirection,
    neighbor_lesson_id: int,
) -> LessonQueueMoveView:
    lesson = _profile_lesson(db, lesson_id)
    neighbor = _profile_lesson(db, neighbor_lesson_id)
    if lesson.id == neighbor.id:
        raise LessonQueueConflictError("a lesson cannot be its own queue neighbor")

    stored_action_id = str(action_id)
    existing = db.scalar(
        select(LessonQueueMoveAction).where(LessonQueueMoveAction.action_id == stored_action_id)
    )
    if existing is not None:
        return _matching_move(
            existing,
            lesson_id=lesson.id,
            direction=direction,
            neighbor_lesson_id=neighbor.id,
        )

    ready_ids = ready_lesson_ids(db)
    try:
        lesson_index = ready_ids.index(lesson.id)
    except ValueError as error:
        raise LessonQueueConflictError("only Ready lessons can be reordered") from error
    expected_index = lesson_index - 1 if direction == "up" else lesson_index + 1
    if not 0 <= expected_index < len(ready_ids):
        raise LessonQueueConflictError(f"lesson cannot move {direction}")
    if ready_ids[expected_index] != neighbor.id:
        raise LessonQueueConflictError(
            f"lesson is no longer adjacent to the requested {direction} neighbor"
        )

    action = LessonQueueMoveAction(
        action_id=stored_action_id,
        lesson_id=lesson.id,
        neighbor_lesson_id=neighbor.id,
        direction=direction,
        occurred_at=utc_now(),
    )
    db.add(action)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        concurrent = db.scalar(
            select(LessonQueueMoveAction).where(LessonQueueMoveAction.action_id == stored_action_id)
        )
        if concurrent is None:
            raise
        return _matching_move(
            concurrent,
            lesson_id=lesson.id,
            direction=direction,
            neighbor_lesson_id=neighbor.id,
        )
    db.refresh(action)
    return _move_view(action)


def _profile_lesson(db: Session, lesson_id: int) -> Lesson:
    profile = db.get(Profile, 1)
    if profile is None:
        raise LookupError("profile is not configured")
    lesson = db.scalar(
        select(Lesson).where(
            Lesson.id == lesson_id,
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
        )
    )
    if lesson is None:
        raise LookupError(f"lesson not found: {lesson_id}")
    return lesson


def _matching_action(
    action: LessonQueueAction, *, lesson_id: int, skipped: bool
) -> LessonQueueActionView:
    if action.lesson_id != lesson_id or action.skipped is not skipped:
        raise LessonQueueConflictError("action_id reused with different queue action data")
    return _action_view(action)


def _action_view(action: LessonQueueAction) -> LessonQueueActionView:
    return LessonQueueActionView(
        action_id=UUID(action.action_id),
        lesson_id=action.lesson_id,
        skipped=action.skipped,
        occurred_at=as_utc(action.occurred_at),
    )


def _matching_move(
    action: LessonQueueMoveAction,
    *,
    lesson_id: int,
    direction: QueueMoveDirection,
    neighbor_lesson_id: int,
) -> LessonQueueMoveView:
    if (
        action.lesson_id != lesson_id
        or action.direction != direction
        or action.neighbor_lesson_id != neighbor_lesson_id
    ):
        raise LessonQueueConflictError("action_id reused with different queue move data")
    return _move_view(action)


def _move_view(action: LessonQueueMoveAction) -> LessonQueueMoveView:
    return LessonQueueMoveView(
        action_id=UUID(action.action_id),
        lesson_id=action.lesson_id,
        direction=cast(QueueMoveDirection, action.direction),
        neighbor_lesson_id=action.neighbor_lesson_id,
        occurred_at=as_utc(action.occurred_at),
    )
