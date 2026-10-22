"""Small UTC helpers shared by persistence projections."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def optional_utc(value: datetime | None) -> datetime | None:
    return as_utc(value) if value is not None else None


def earliest_datetime(current: datetime | None, candidate: datetime) -> datetime:
    return candidate if current is None or candidate < current else current


def latest_datetime(current: datetime | None, candidate: datetime) -> datetime:
    return candidate if current is None or candidate > current else current
