"""Typed access to durable generation-task identity and intent."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, cast

from server.clock import as_utc
from server.models import GenerationTask
from server.schemas import GenerationRequestKind

GenerationMode = Literal["lesson", "calibration"]


def generation_task_kind(task: GenerationTask) -> GenerationRequestKind:
    value = task.payload.get("request_kind", "queue_fill")
    if value not in {"queue_fill", "topic_request"}:
        raise ValueError("generation task has an invalid request kind")
    return cast(GenerationRequestKind, value)


def generation_task_mode(task: GenerationTask) -> GenerationMode:
    value = task.payload.get("generation_mode", "lesson")
    if value not in {"lesson", "calibration"}:
        raise ValueError("generation task has an invalid mode")
    return cast(GenerationMode, value)


def requested_topic(task: GenerationTask) -> str | None:
    if generation_task_kind(task) == "queue_fill":
        return None
    value = task.payload.get("requested_topic")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("topic request generation task has an invalid requested topic")
    return value


def topic_request_id(task: GenerationTask) -> str | None:
    if generation_task_kind(task) == "queue_fill":
        return None
    value = task.payload.get("request_id")
    if not isinstance(value, str) or not value:
        raise ValueError("topic request generation task has an invalid request ID")
    return value


def generation_queue_target(task: GenerationTask) -> int:
    if generation_task_kind(task) != "queue_fill":
        raise ValueError("topic request generation tasks do not have a queue target")
    value = task.payload.get("queue_target")
    if not isinstance(value, int) or not 1 <= value <= 10:
        raise ValueError("generation task has an invalid queue target")
    return value


def waiting_for_model_capacity(task: GenerationTask) -> bool:
    """True while the latest transition was an admission deferral that has not been retried."""
    retry_at = task.payload.get("admission_retry_at")
    if task.state != "pending" or not isinstance(retry_at, str):
        return False
    return as_utc(datetime.fromisoformat(retry_at)) > as_utc(task.updated_at)


def generated_lesson_key(task_id: int, index: int) -> str:
    if task_id <= 0 or index <= 0:
        raise ValueError("generated lesson identity requires positive task and lesson indexes")
    return f"generated-task-{task_id}-{index}"
