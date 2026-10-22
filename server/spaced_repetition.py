"""Shared stability math for evidence-based learning estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class StabilityPolicy:
    """Coefficients controlling how quickly memory stability changes."""

    recall_at_stability: float = 0.9
    minimum_days: float = 0.1
    maximum_days: float = 365.0
    success_forgetting_bonus: float = 1.65
    failure_base: float = 0.35
    failure_recall_penalty: float = 0.45

    def __post_init__(self) -> None:
        if not 0 < self.recall_at_stability <= 1:
            raise ValueError("recall_at_stability must be between 0 and 1")
        if (
            not math.isfinite(self.minimum_days)
            or self.minimum_days <= 0
            or not math.isfinite(self.maximum_days)
            or self.maximum_days < self.minimum_days
        ):
            raise ValueError("stability bounds must be finite, positive, and ordered")
        coefficients = (
            self.success_forgetting_bonus,
            self.failure_base,
            self.failure_recall_penalty,
        )
        if any(not math.isfinite(value) or value < 0 for value in coefficients):
            raise ValueError("stability coefficients must be finite and non-negative")


DEFAULT_STABILITY_POLICY = StabilityPolicy()


def estimated_recall(
    *,
    mastery: float,
    stability_days: float,
    last_seen_at: datetime | None,
    at: datetime,
    policy: StabilityPolicy = DEFAULT_STABILITY_POLICY,
) -> float:
    """Estimate recall now from mastery, stability, and elapsed time."""

    if last_seen_at is None:
        return mastery
    elapsed_days = max(0.0, (at - last_seen_at).total_seconds() / SECONDS_PER_DAY)
    return float(
        policy.recall_at_stability ** (elapsed_days / max(stability_days, policy.minimum_days))
    )


def stability_after_weighted_success(
    *,
    stability_days: float,
    recall: float,
    evidence_mass: float,
    gain_days_per_mass: float,
    policy: StabilityPolicy = DEFAULT_STABILITY_POLICY,
) -> float:
    """Add interpretable stability days, rewarding a successful delayed retrieval."""

    if any(not math.isfinite(value) or value < 0 for value in (evidence_mass, gain_days_per_mass)):
        raise ValueError("success evidence and stability gain must be finite and non-negative")
    bounded_recall = min(1.0, max(0.0, recall))
    delay_multiplier = 1.0 + policy.success_forgetting_bonus * (1.0 - bounded_recall)
    gain = gain_days_per_mass * evidence_mass * delay_multiplier
    return _bounded(stability_days + gain, policy)


def stability_after_failure(
    *,
    stability_days: float,
    recall: float,
    penalty: float,
    policy: StabilityPolicy = DEFAULT_STABILITY_POLICY,
) -> float:
    """Reduce stability after a weighted failure."""

    factor = 1.0 - penalty * (policy.failure_base + policy.failure_recall_penalty * recall)
    return _bounded(stability_days * factor, policy)


def _bounded(days: float, policy: StabilityPolicy) -> float:
    return min(policy.maximum_days, max(policy.minimum_days, days))
