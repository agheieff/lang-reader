from __future__ import annotations

from datetime import datetime, timedelta, timezone

from server.clock import as_utc, latest_datetime, optional_utc, utc_now


def test_clock_helpers_normalize_and_compare_datetimes() -> None:
    naive = datetime(2026, 1, 1, 12)
    offset = datetime(2026, 1, 1, 13, tzinfo=timezone(timedelta(hours=1)))

    assert as_utc(naive) == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    assert as_utc(offset) == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    assert optional_utc(None) is None
    assert latest_datetime(None, naive) is naive
    assert latest_datetime(naive, naive + timedelta(minutes=1)) > naive
    assert utc_now().tzinfo is timezone.utc
