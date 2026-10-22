from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from server.spaced_repetition import (
    StabilityPolicy,
    estimated_recall,
    stability_after_failure,
    stability_after_weighted_success,
)


def test_recall_starts_at_mastery_and_decays_to_policy_anchor() -> None:
    seen_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert estimated_recall(
        mastery=0.7,
        stability_days=4.0,
        last_seen_at=None,
        at=seen_at,
    ) == pytest.approx(0.7)
    assert estimated_recall(
        mastery=0.7,
        stability_days=4.0,
        last_seen_at=seen_at,
        at=seen_at + timedelta(days=4),
    ) == pytest.approx(0.9)


def test_success_and_failure_obey_configured_bounds() -> None:
    policy = StabilityPolicy(minimum_days=0.2, maximum_days=10.0)

    assert stability_after_weighted_success(
        stability_days=10.0,
        recall=0.1,
        evidence_mass=1.0,
        gain_days_per_mass=1.0,
        policy=policy,
    ) == pytest.approx(10.0)
    assert stability_after_failure(
        stability_days=0.2,
        recall=1.0,
        penalty=1.0,
        policy=policy,
    ) == pytest.approx(0.2)


def test_weighted_success_uses_interpretable_days_and_rewards_delay() -> None:
    immediate = stability_after_weighted_success(
        stability_days=1.0,
        recall=1.0,
        evidence_mass=1.5,
        gain_days_per_mass=0.5,
    )
    delayed = stability_after_weighted_success(
        stability_days=1.0,
        recall=0.5,
        evidence_mass=1.5,
        gain_days_per_mass=0.5,
    )

    assert immediate == pytest.approx(1.75)
    assert delayed > immediate


@pytest.mark.parametrize(
    "policy",
    [
        {"recall_at_stability": 0.0},
        {"minimum_days": 0.0},
        {"minimum_days": 2.0, "maximum_days": 1.0},
        {"failure_base": -0.1},
    ],
)
def test_stability_policy_rejects_invalid_configuration(policy: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        StabilityPolicy(**policy)
