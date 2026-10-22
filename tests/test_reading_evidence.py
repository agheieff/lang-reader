from __future__ import annotations

import pytest

from server.reading_evidence import (
    ReadingEvidencePolicy,
    is_qualified_reading,
    payload_is_qualified_reading,
)


@pytest.mark.parametrize(
    ("active_seconds", "completion_ratio", "expected"),
    [
        (30, 0.8, True),
        (29.9, 1.0, False),
        (60, 0.799, False),
        (True, 1.0, False),
        (60, float("nan"), False),
        (60, 1.01, False),
    ],
)
def test_qualified_reading_requires_finite_meaningful_progress(
    active_seconds: object,
    completion_ratio: object,
    expected: bool,
) -> None:
    assert (
        is_qualified_reading(
            active_seconds=active_seconds,
            completion_ratio=completion_ratio,
        )
        is expected
    )


def test_payload_qualification_accepts_an_explicit_policy() -> None:
    policy = ReadingEvidencePolicy(
        minimum_active_seconds=10.0,
        minimum_completion_ratio=0.5,
    )

    assert payload_is_qualified_reading(
        {"active_seconds": 10, "completion_ratio": 0.5},
        policy,
    )


@pytest.mark.parametrize(
    "policy",
    [
        {"minimum_active_seconds": -1},
        {"minimum_completion_ratio": 1.1},
    ],
)
def test_reading_policy_rejects_invalid_configuration(policy: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ReadingEvidencePolicy(**policy)
