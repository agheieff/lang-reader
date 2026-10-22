"""Durable lesson-generation tasks: enqueue, trigger tracking, claim, retry, failure."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server import settings
from server.clock import as_utc, utc_now
from server.derived_state import ensure_derived_states
from server.generation_tasks import (
    GenerationMode,
    generation_task_kind,
    generation_task_mode,
    requested_topic,
    topic_request_id,
)
from server.lesson_content import lesson_document
from server.lesson_queue import (
    ready_lessons,
    unread_lessons,
)
from server.models import (
    GenerationTask,
    Interaction,
    LessonQueueAction,
)
from server.profile_activation import (
    profile_is_active,
)
from server.profile_fingerprint import profile_fingerprint
from server.profile_state import (
    ProfileInactiveError,
    ensure_profile,
    generation_mode,
    profile_level_source,
    workspace_key_and_path,
)
from server.schemas import (
    GenerationFailureContext,
    TextRequestIn,
)
from server.session_feedback import latest_profile_feedback

GENERATION_DEDUPE_KEY = "lesson-queue"
GENERATION_CONTRACT_REVISION = 24
DEFAULT_UNREAD_QUEUE_TARGET = 3
MAX_GENERATION_ATTEMPTS = 2
MAX_STAGED_GENERATION_ATTEMPTS = 6
MAX_GENERATION_STAGE_FAILURES = 2
MAX_GENERATION_FAILURE_CONTEXT = 4
MAX_GENERATION_TRIGGER_IDS = 64
GENERATION_FAILURE_STAGES = (
    "prose",
    "lexical",
    "translation",
    "grammar",
    "assembly",
    "legacy",
)
GENERATION_TRIGGER_EVENT_TYPES = (
    "lesson.started",
    "term.revealed",
    "translation.revealed",
    "lesson.completed",
    "lesson.rated",
)
MAX_LESSONS_PER_GENERATION_TASK = 1
CALIBRATION_QUEUE_TARGET = 1


@dataclass(frozen=True)
class _GenerationTriggers:
    event_ids: tuple[str, ...] = ()
    queue_action_ids: tuple[str, ...] = ()
    event_cursor: int = 0
    queue_action_cursor: int = 0


class TextRequestConflictError(ValueError):
    pass


def auto_generation_enabled() -> bool:
    """Enabled unless explicitly disabled, matching the documented `pnpm dev` default."""

    return settings.load().auto_generate


def unread_queue_target() -> int:
    return settings.load().queue_target


def unread_generation_lesson_count(db: Session, mode: GenerationMode) -> int:
    candidates = unread_lessons(db) if mode == "calibration" else ready_lessons(db)
    return sum(
        lesson_document(lesson).calibration is not None
        if mode == "calibration"
        else lesson_document(lesson).calibration is None
        for lesson in candidates
    )


def request_topic_lesson(db: Session, request: TextRequestIn) -> GenerationTask:
    """Durably enqueue one extra lesson, independently of the automatic unread target."""

    profile = ensure_profile(db)
    if not profile_is_active(profile):
        raise ProfileInactiveError("activate this language before requesting a text")
    request_id = str(request.request_id)
    dedupe_key = f"topic-request:{request_id}"
    existing = db.scalar(
        select(GenerationTask).where(GenerationTask.dedupe_key == dedupe_key).limit(1)
    )
    if existing is not None:
        return _matching_topic_request(existing, request_id=request_id, topic=request.topic)

    key, workspace_path = workspace_key_and_path(db, profile)
    task = GenerationTask(
        state="pending",
        dedupe_key=dedupe_key,
        payload={
            "schema_version": 1,
            "generation_contract_revision": GENERATION_CONTRACT_REVISION,
            "request_kind": "topic_request",
            "request_id": request_id,
            "requested_topic": request.topic,
            "needed_lesson_count": 1,
            "generation_mode": "lesson",
            "learning_language": profile.learning_language,
            "translation_language": profile.translation_language,
            "profile_key": key,
            "workspace_path": workspace_path,
            "profile_fingerprint": profile_fingerprint(db, profile),
            "trigger_event_ids": [],
            "latest_feedback": latest_profile_feedback(db),
        },
    )
    db.add(task)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        concurrent = db.scalar(
            select(GenerationTask).where(GenerationTask.dedupe_key == dedupe_key).limit(1)
        )
        if concurrent is None:
            raise
        return _matching_topic_request(concurrent, request_id=request_id, topic=request.topic)
    db.refresh(task)
    return task


def _matching_topic_request(task: GenerationTask, *, request_id: str, topic: str) -> GenerationTask:
    if (
        generation_task_kind(task) != "topic_request"
        or topic_request_id(task) != request_id
        or requested_topic(task) != topic
    ):
        raise TextRequestConflictError("request_id reused with different topic request data")
    return task


def generation_failure_context(task: GenerationTask) -> list[GenerationFailureContext]:
    """Return bounded, validated diagnostics from earlier attempts and legacy task rows."""

    failures: list[GenerationFailureContext] = []
    raw_failures = task.payload.get("previous_failures")
    if isinstance(raw_failures, list):
        for raw_failure in raw_failures:
            try:
                failure = GenerationFailureContext.model_validate(raw_failure)
            except ValueError:
                continue
            if failure not in failures:
                failures.append(failure)
    if task.state == "failed" and isinstance(task.error, str) and task.error:
        legacy = GenerationFailureContext(
            task_id=task.id,
            attempt=max(1, task.attempts),
            error=task.error[:4000],
        )
        if legacy not in failures:
            failures.append(legacy)
    return failures[-MAX_GENERATION_FAILURE_CONTEXT:]


def _generation_failure_buckets(error: str) -> tuple[str, ...]:
    stages = tuple(stage for stage in GENERATION_FAILURE_STAGES if f"[{stage}]" in error)
    return stages or ("task",)


def _generation_stage_failure_counts(task: GenerationTask) -> Counter[str]:
    raw_counts = task.payload.get("generation_stage_failures")
    if isinstance(raw_counts, dict):
        stored_counts = Counter(
            {
                stage: count
                for stage, count in raw_counts.items()
                if (
                    isinstance(stage, str)
                    and (stage in GENERATION_FAILURE_STAGES or stage == "task")
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and count > 0
                )
            }
        )
        if stored_counts:
            return stored_counts

    derived_counts: Counter[str] = Counter()
    for failure in generation_failure_context(task):
        if failure.task_id == task.id:
            derived_counts.update(_generation_failure_buckets(failure.error))
    return derived_counts


def _generation_attempt_limit(task: GenerationTask) -> int:
    if generation_task_mode(task) == "calibration":
        return MAX_GENERATION_ATTEMPTS
    return MAX_STAGED_GENERATION_ATTEMPTS


def _generation_retry_available(task: GenerationTask) -> bool:
    # Dispatch numbers name immutable artifact directories. Capacity-only waits do not spend
    # the draft failure budget, but must not cause the next dispatch to overwrite evidence.
    if task.attempts - int(task.payload.get("admission_deferrals", 0)) >= _generation_attempt_limit(
        task
    ):
        return False
    if task.payload.get("generation_contract_revision") != GENERATION_CONTRACT_REVISION:
        return True
    counts = _generation_stage_failure_counts(task)
    return not counts or max(counts.values()) < MAX_GENERATION_STAGE_FAILURES


def _append_generation_failure(task: GenerationTask, error: str) -> str:
    stored_error = error[:4000] or "generation failed without an error message"
    failures = generation_failure_context(task)
    current_failure = GenerationFailureContext(
        task_id=task.id,
        attempt=max(1, task.attempts),
        error=stored_error,
    )
    if current_failure not in failures:
        failures.append(current_failure)
    payload = dict(task.payload)
    counts = _generation_stage_failure_counts(task)
    if current_failure not in generation_failure_context(task):
        counts.update(_generation_failure_buckets(stored_error))
    payload["generation_stage_failures"] = dict(sorted(counts.items()))
    payload["previous_failures"] = [
        failure.model_dump(mode="json") for failure in failures[-MAX_GENERATION_FAILURE_CONTEXT:]
    ]
    task.payload = payload
    return stored_error


def _resolve_generation_triggers(
    db: Session,
    *,
    event_ids: Sequence[str] = (),
    queue_action_ids: Sequence[str] = (),
    require_all: bool = True,
) -> _GenerationTriggers:
    normalized_event_ids = _normalized_trigger_ids(event_ids)
    normalized_action_ids = _normalized_trigger_ids(queue_action_ids)
    event_rows = {
        event_id: row_id
        for event_id, row_id in db.execute(
            select(Interaction.event_id, Interaction.id).where(
                Interaction.event_id.in_(normalized_event_ids)
            )
        ).all()
    }
    action_rows = {
        action_id: row_id
        for action_id, row_id in db.execute(
            select(LessonQueueAction.action_id, LessonQueueAction.id).where(
                LessonQueueAction.action_id.in_(normalized_action_ids)
            )
        ).all()
    }
    if require_all:
        missing_events = set(normalized_event_ids) - event_rows.keys()
        missing_actions = set(normalized_action_ids) - action_rows.keys()
        if missing_events or missing_actions:
            raise ValueError("generation triggers must reference accepted events or queue actions")
    return _GenerationTriggers(
        event_ids=tuple(event_id for event_id in normalized_event_ids if event_id in event_rows),
        queue_action_ids=tuple(
            action_id for action_id in normalized_action_ids if action_id in action_rows
        ),
        event_cursor=max(event_rows.values(), default=0),
        queue_action_cursor=max(action_rows.values(), default=0),
    )


def _normalized_trigger_ids(values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("generation trigger IDs must be non-empty strings")
        if value not in normalized:
            normalized.append(value)
    return normalized


def _task_generation_triggers(
    db: Session, task: GenerationTask, *, deferred: bool = False
) -> _GenerationTriggers:
    prefix = "deferred_" if deferred else ""
    event_ids = _payload_trigger_ids(task.payload.get(f"{prefix}trigger_event_ids"))
    action_ids = _payload_trigger_ids(task.payload.get(f"{prefix}trigger_queue_action_ids"))
    resolved = _resolve_generation_triggers(
        db,
        event_ids=event_ids,
        queue_action_ids=action_ids,
        require_all=False,
    )
    return _GenerationTriggers(
        event_ids=resolved.event_ids,
        queue_action_ids=resolved.queue_action_ids,
        event_cursor=max(
            resolved.event_cursor,
            _payload_trigger_cursor(task.payload.get(f"{prefix}trigger_event_cursor")),
        ),
        queue_action_cursor=max(
            resolved.queue_action_cursor,
            _payload_trigger_cursor(task.payload.get(f"{prefix}trigger_queue_action_cursor")),
        ),
    )


def _payload_trigger_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _payload_trigger_cursor(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _combine_generation_triggers(*groups: _GenerationTriggers) -> _GenerationTriggers:
    event_ids = _bounded_trigger_ids(item for group in groups for item in group.event_ids)
    action_ids = _bounded_trigger_ids(item for group in groups for item in group.queue_action_ids)
    return _GenerationTriggers(
        event_ids=event_ids,
        queue_action_ids=action_ids,
        event_cursor=max((group.event_cursor for group in groups), default=0),
        queue_action_cursor=max((group.queue_action_cursor for group in groups), default=0),
    )


def _bounded_trigger_ids(values: Iterable[str]) -> tuple[str, ...]:
    unique: list[str] = []
    for value in values:
        if value in unique:
            unique.remove(value)
        unique.append(value)
    return tuple(unique[-MAX_GENERATION_TRIGGER_IDS:])


def _write_task_generation_triggers(
    task: GenerationTask, triggers: _GenerationTriggers, *, deferred: bool = False
) -> bool:
    prefix = "deferred_" if deferred else ""
    payload = dict(task.payload)
    values: dict[str, object] = {
        f"{prefix}trigger_event_ids": list(triggers.event_ids),
        f"{prefix}trigger_queue_action_ids": list(triggers.queue_action_ids),
        f"{prefix}trigger_event_cursor": triggers.event_cursor,
        f"{prefix}trigger_queue_action_cursor": triggers.queue_action_cursor,
    }
    if all(payload.get(key) == value for key, value in values.items()):
        return False
    payload.update(values)
    task.payload = payload
    return True


def _merge_task_generation_triggers(
    db: Session,
    task: GenerationTask,
    incoming: _GenerationTriggers,
    *,
    deferred: bool,
) -> bool:
    current = _task_generation_triggers(db, task, deferred=deferred)
    return _write_task_generation_triggers(
        task,
        _combine_generation_triggers(current, incoming),
        deferred=deferred,
    )


def _has_fresh_generation_trigger(
    candidate: _GenerationTriggers, consumed: _GenerationTriggers
) -> bool:
    return (
        candidate.event_cursor > consumed.event_cursor
        or candidate.queue_action_cursor > consumed.queue_action_cursor
    )


def _generation_trigger_high_water(db: Session, after: _GenerationTriggers) -> _GenerationTriggers:
    """Return committed generation evidence newer than a task's durable cursors."""

    event_rows = db.execute(
        select(Interaction.event_id, Interaction.id)
        .where(
            Interaction.event_type.in_(GENERATION_TRIGGER_EVENT_TYPES),
            Interaction.id > after.event_cursor,
        )
        .order_by(Interaction.id.desc())
        .limit(MAX_GENERATION_TRIGGER_IDS)
    ).all()
    action_rows = db.execute(
        select(LessonQueueAction.action_id, LessonQueueAction.id)
        .where(
            LessonQueueAction.skipped.is_(True),
            LessonQueueAction.id > after.queue_action_cursor,
        )
        .order_by(LessonQueueAction.id.desc())
        .limit(MAX_GENERATION_TRIGGER_IDS)
    ).all()
    return _GenerationTriggers(
        event_ids=tuple(row.event_id for row in reversed(event_rows)),
        queue_action_ids=tuple(row.action_id for row in reversed(action_rows)),
        event_cursor=event_rows[0].id if event_rows else 0,
        queue_action_cursor=action_rows[0].id if action_rows else 0,
    )


def ensure_generation_task(
    db: Session,
    *,
    trigger_event_ids: Sequence[str] = (),
    trigger_queue_action_ids: Sequence[str] = (),
    require_enabled: bool = True,
    proficiency_current: bool = False,
) -> GenerationTask | None:
    if require_enabled and not auto_generation_enabled():
        return None

    profile = ensure_profile(db)
    if not profile_is_active(profile):
        return None
    if not proficiency_current and profile_level_source(profile) == "unknown":
        # Finishing calibration placement can change the generation mode itself.
        ensure_derived_states(db)
    mode = generation_mode(profile)
    target = CALIBRATION_QUEUE_TARGET if mode == "calibration" else unread_queue_target()
    unread = unread_generation_lesson_count(db, mode)
    if unread >= target:
        return None

    explicit_triggers = _resolve_generation_triggers(
        db,
        event_ids=trigger_event_ids,
        queue_action_ids=trigger_queue_action_ids,
    )

    outstanding = db.scalar(
        select(GenerationTask)
        .where(
            GenerationTask.dedupe_key == GENERATION_DEDUPE_KEY,
            GenerationTask.state.in_(("pending", "running")),
        )
        .order_by(GenerationTask.id)
        .limit(1)
    )
    if outstanding is not None:
        baseline = _combine_generation_triggers(
            _task_generation_triggers(db, outstanding),
            _task_generation_triggers(db, outstanding, deferred=True),
        )
        observed_triggers = _combine_generation_triggers(
            _generation_trigger_high_water(db, baseline),
            explicit_triggers,
        )
        changed = _merge_task_generation_triggers(
            db,
            outstanding,
            observed_triggers,
            deferred=outstanding.state == "running",
        )
        if changed:
            db.commit()
            db.refresh(outstanding)
        return outstanding
    if not proficiency_current:
        ensure_derived_states(db)
    latest = next(
        (
            task
            for task in db.scalars(select(GenerationTask).order_by(GenerationTask.id.desc())).all()
            if generation_task_kind(task) == "queue_fill"
        ),
        None,
    )
    baseline = (
        _combine_generation_triggers(
            _task_generation_triggers(db, latest),
            _task_generation_triggers(db, latest, deferred=True),
        )
        if latest is not None
        else _GenerationTriggers()
    )
    observed_triggers = _combine_generation_triggers(
        _generation_trigger_high_water(db, baseline),
        explicit_triggers,
    )
    fingerprint = profile_fingerprint(db, profile)
    task_triggers = observed_triggers
    previous_failures: list[GenerationFailureContext] = []
    if latest is not None and latest.state == "failed":
        consumed = _task_generation_triggers(db, latest)
        deferred = _task_generation_triggers(db, latest, deferred=True)
        same_fingerprint = latest.payload.get("profile_fingerprint") == fingerprint
        same_contract = (
            latest.payload.get("generation_contract_revision") == GENERATION_CONTRACT_REVISION
        )
        if same_fingerprint:
            previous_failures = generation_failure_context(latest)
        if same_fingerprint and same_contract:
            if _generation_retry_available(latest):
                if _merge_task_generation_triggers(
                    db,
                    latest,
                    observed_triggers,
                    deferred=False,
                ):
                    db.commit()
                    db.refresh(latest)
                return latest
            unconsumed = _combine_generation_triggers(deferred, observed_triggers)
            if not _has_fresh_generation_trigger(unconsumed, consumed):
                return latest
        task_triggers = _combine_generation_triggers(consumed, deferred, observed_triggers)

    key, workspace_path = workspace_key_and_path(db, profile)
    task = GenerationTask(
        state="pending",
        dedupe_key=GENERATION_DEDUPE_KEY,
        payload={
            "schema_version": 1,
            "generation_contract_revision": GENERATION_CONTRACT_REVISION,
            "request_kind": "queue_fill",
            "queue_target": target,
            "unread_lesson_count": unread,
            "needed_lesson_count": min(MAX_LESSONS_PER_GENERATION_TASK, target - unread),
            "queue_shortfall": target - unread,
            "generation_mode": mode,
            "learning_language": profile.learning_language,
            "translation_language": profile.translation_language,
            "profile_key": key,
            "workspace_path": workspace_path,
            "profile_fingerprint": fingerprint,
            "trigger_event_ids": list(task_triggers.event_ids),
            "trigger_queue_action_ids": list(task_triggers.queue_action_ids),
            "trigger_event_cursor": task_triggers.event_cursor,
            "trigger_queue_action_cursor": task_triggers.queue_action_cursor,
            "previous_failures": [failure.model_dump(mode="json") for failure in previous_failures],
            "latest_feedback": latest_profile_feedback(db),
        },
    )
    db.add(task)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return db.scalar(
            select(GenerationTask)
            .where(
                GenerationTask.dedupe_key == GENERATION_DEDUPE_KEY,
                GenerationTask.state.in_(("pending", "running")),
            )
            .order_by(GenerationTask.id)
            .limit(1)
        )
    db.refresh(task)
    return task


def maintain_generation_task(db: Session) -> GenerationTask | None:
    """Fill the queue or retry the failed DAG stage within its bounded repair budget."""

    if not profile_is_active(ensure_profile(db)):
        return None
    task = ensure_generation_task(db, require_enabled=False)
    if task is not None and task.state == "failed" and _generation_retry_available(task):
        return retry_generation_task(db, task.id)
    return task


def maintain_topic_generation_tasks(db: Session) -> list[GenerationTask]:
    """Retry eligible failed explicit requests without serializing them with queue work."""

    if not profile_is_active(ensure_profile(db)):
        return []
    retried: list[GenerationTask] = []
    tasks = db.scalars(
        select(GenerationTask)
        .where(GenerationTask.state == "failed")
        .order_by(GenerationTask.created_at, GenerationTask.id)
    ).all()
    for task in tasks:
        if generation_task_kind(task) == "topic_request" and _generation_retry_available(task):
            retried.append(retry_generation_task(db, task.id))
    return retried


def claim_generation_task(db: Session) -> GenerationTask | None:
    if not profile_is_active(ensure_profile(db)):
        return None
    while True:
        task = db.scalar(
            select(GenerationTask)
            .where(GenerationTask.state == "pending")
            .order_by(GenerationTask.created_at, GenerationTask.id)
            .limit(1)
        )
        if task is None:
            return None
        retry_at = task.payload.get("admission_retry_at")
        if isinstance(retry_at, str) and utc_now() < as_utc(datetime.fromisoformat(retry_at)):
            return None
        if _generation_retry_available(task):
            break
        now = utc_now()
        task.state = "failed"
        if generation_task_kind(task) == "queue_fill":
            task.dedupe_key = None
        task.error = _append_generation_failure(
            task,
            task.error or "generation task reached the maximum attempt count before claim",
        )
        task.started_at = None
        task.finished_at = now
        task.updated_at = now
        db.commit()
    now = utc_now()
    task.state = "running"
    task.attempts += 1
    task.started_at = now
    task.finished_at = None
    task.updated_at = now
    task.error = None
    db.commit()
    db.refresh(task)
    return task


def defer_generation_task(
    db: Session, task_id: int, *, log_path: str, reason: str
) -> GenerationTask:
    """Retain a dispatch refused by admission, preserving identity and validated stage files."""
    task = _running_generation_task(db, task_id)
    now = utc_now()
    task.payload = {
        **task.payload,
        "admission_deferrals": int(task.payload.get("admission_deferrals", 0)) + 1,
        "admission_retry_at": (now + timedelta(seconds=60)).isoformat(),
    }
    task.state = "pending"
    task.log_path = log_path
    task.error = reason
    task.started_at = None
    task.finished_at = None
    task.updated_at = now
    db.commit()
    db.refresh(task)
    return task


def recover_running_generation_tasks(db: Session) -> int:
    tasks = db.scalars(select(GenerationTask).where(GenerationTask.state == "running")).all()
    now = utc_now()
    for task in tasks:
        task.error = _append_generation_failure(
            task,
            "worker stopped before reporting a result",
        )
        terminal = not _generation_retry_available(task)
        task.state = "failed" if terminal else "pending"
        if terminal and generation_task_kind(task) == "queue_fill":
            task.dedupe_key = None
        task.started_at = None
        task.finished_at = now if terminal else None
        task.updated_at = now
    db.commit()
    return len(tasks)


def complete_generation_task(db: Session, task_id: int, *, log_path: str) -> GenerationTask:
    task = _running_generation_task(db, task_id)
    now = utc_now()
    task.state = "completed"
    if generation_task_kind(task) == "queue_fill":
        task.dedupe_key = None
    task.log_path = log_path
    task.error = None
    task.finished_at = now
    task.updated_at = now
    db.commit()
    db.refresh(task)
    return task


def fail_generation_task(db: Session, task_id: int, *, log_path: str, error: str) -> GenerationTask:
    task = _running_generation_task(db, task_id)
    now = utc_now()
    stored_error = _append_generation_failure(task, error)
    task.state = "failed"
    if generation_task_kind(task) == "queue_fill":
        task.dedupe_key = None
    task.log_path = log_path
    task.error = stored_error
    task.finished_at = now
    task.updated_at = now
    db.commit()
    db.refresh(task)
    return task


def retry_generation_task(db: Session, task_id: int) -> GenerationTask:
    if not profile_is_active(ensure_profile(db)):
        raise ProfileInactiveError("activate this language before retrying generation")
    task = db.get(GenerationTask, task_id)
    if task is None:
        raise LookupError(f"generation task not found: {task_id}")
    if task.state != "failed":
        raise ValueError("only failed generation tasks can be retried")
    if not _generation_retry_available(task):
        raise ValueError("generation task reached the maximum attempt count for this stage")
    now = utc_now()
    contract_changed = (
        task.payload.get("generation_contract_revision") != GENERATION_CONTRACT_REVISION
    )
    failures = [] if contract_changed else generation_failure_context(task)
    if contract_changed:
        payload = dict(task.payload)
        payload["generation_contract_revision"] = GENERATION_CONTRACT_REVISION
        payload["generation_stage_failures"] = {}
        payload["previous_failures"] = []
        task.payload = payload
    elif failures:
        payload = dict(task.payload)
        payload["previous_failures"] = [
            failure.model_dump(mode="json")
            for failure in failures[-MAX_GENERATION_FAILURE_CONTEXT:]
        ]
        task.payload = payload
    active_triggers = _task_generation_triggers(db, task)
    deferred_triggers = _task_generation_triggers(db, task, deferred=True)
    _write_task_generation_triggers(
        task,
        _combine_generation_triggers(active_triggers, deferred_triggers),
    )
    _write_task_generation_triggers(task, _GenerationTriggers(), deferred=True)
    task.state = "pending"
    task.dedupe_key = _generation_task_dedupe_key(task)
    task.log_path = None
    task.error = None
    task.started_at = None
    task.finished_at = None
    task.updated_at = now
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ValueError("another generation task has the same durable identity") from error
    db.refresh(task)
    return task


def _generation_task_dedupe_key(task: GenerationTask) -> str:
    if generation_task_kind(task) == "queue_fill":
        return GENERATION_DEDUPE_KEY
    request_id = topic_request_id(task)
    if request_id is None:
        raise ValueError("topic request task has an invalid request ID")
    return f"topic-request:{request_id}"


def _running_generation_task(db: Session, task_id: int) -> GenerationTask:
    task = db.get(GenerationTask, task_id)
    if task is None:
        raise LookupError(f"generation task not found: {task_id}")
    if task.state != "running":
        raise ValueError(f"generation task {task_id} is not running")
    return task


def queue_generation_status(db: Session) -> Literal["pending", "running", "failed"] | None:
    latest = db.scalar(select(GenerationTask).order_by(GenerationTask.id.desc()).limit(1))
    if latest is None or latest.state == "completed":
        return None
    return latest.state
