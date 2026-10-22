"""Shared policy for deciding whether a reading session contains usable evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from server.clock import as_utc
from server.models import Interaction


@dataclass(frozen=True)
class ReadingEvidencePolicy:
    minimum_active_seconds: float = 30.0
    minimum_completion_ratio: float = 0.8

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_active_seconds) or self.minimum_active_seconds < 0:
            raise ValueError("minimum_active_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.minimum_completion_ratio)
            or not 0 <= self.minimum_completion_ratio <= 1
        ):
            raise ValueError("minimum_completion_ratio must be between 0 and 1")


DEFAULT_READING_EVIDENCE_POLICY = ReadingEvidencePolicy()


def is_qualified_reading(
    *,
    active_seconds: object,
    completion_ratio: object,
    policy: ReadingEvidencePolicy = DEFAULT_READING_EVIDENCE_POLICY,
) -> bool:
    """Return whether timing and progress are sufficient to infer learning."""

    active = finite_number(active_seconds)
    ratio = finite_number(completion_ratio)
    return (
        active is not None
        and active >= policy.minimum_active_seconds
        and ratio is not None
        and policy.minimum_completion_ratio <= ratio <= 1.0
    )


def payload_is_qualified_reading(
    payload: Mapping[str, object],
    policy: ReadingEvidencePolicy = DEFAULT_READING_EVIDENCE_POLICY,
) -> bool:
    return is_qualified_reading(
        active_seconds=payload.get("active_seconds"),
        completion_ratio=payload.get("completion_ratio"),
        policy=policy,
    )


def qualified_completion_and_considered(
    events: Sequence[Interaction],
    policy: ReadingEvidencePolicy = DEFAULT_READING_EVIDENCE_POLICY,
) -> tuple[Interaction | None, list[Interaction]]:
    """Return a session's first qualified completion and the events at or before it.

    Without a qualified completion every event stays considered, since negative evidence
    (reveals) does not require finishing the lesson.
    """

    completion = next(
        (
            event
            for event in events
            if event.event_type == "lesson.completed"
            and payload_is_qualified_reading(event.payload, policy)
        ),
        None,
    )
    if completion is None:
        return None, list(events)
    cutoff = (as_utc(completion.occurred_at), completion.id)
    return completion, [
        event for event in events if (as_utc(event.occurred_at), event.id) <= cutoff
    ]


def finite_number(value: object) -> float | None:
    """Return a real finite numeric value while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None
