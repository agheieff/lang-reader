"""Small immutable policies shared by learning projections."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TermBandPolicy:
    """Thresholds shared by reader styling and vocabulary statistics."""

    familiar_min_exposures: int = 5
    familiar_min_mastery: float = 0.64
    familiar_with_failures_min_exposures: int = 8
    familiar_with_failures_min_mastery: float = 0.72
    expected_min_exposures: int = 2
    expected_min_mastery: float = 0.56
    mastered_min_mastery: float = 0.85
    mastered_min_exposures: int = 3
    repeated_reveal_threshold: int = 2
    base_expected_frequency_rank: int = 500
    max_expected_frequency_rank: int = 50_000
    uncertain_frequency_multiplier: float = 3.0

    def __post_init__(self) -> None:
        if not (
            0
            <= self.expected_min_mastery
            < self.familiar_min_mastery
            < self.mastered_min_mastery
            <= 1
        ):
            raise ValueError("term mastery thresholds must be ordered values between 0 and 1")
        if not 0 <= self.familiar_with_failures_min_mastery <= 1:
            raise ValueError("failure-adjusted familiarity mastery must be between 0 and 1")
        exposure_counts = (
            self.familiar_min_exposures,
            self.familiar_with_failures_min_exposures,
            self.expected_min_exposures,
            self.mastered_min_exposures,
            self.repeated_reveal_threshold,
        )
        if any(value < 0 for value in exposure_counts):
            raise ValueError("term evidence counts must be non-negative")
        if not 0 < self.base_expected_frequency_rank <= self.max_expected_frequency_rank:
            raise ValueError("frequency-rank bounds must be positive and ordered")
        if self.uncertain_frequency_multiplier < 1:
            raise ValueError("uncertain_frequency_multiplier must be at least 1")


TERM_BAND_POLICY = TermBandPolicy()


def frontier_frequency_rank(
    difficulty: float,
    lower: float | None,
    policy: TermBandPolicy = TERM_BAND_POLICY,
) -> int:
    """Corpus frequency rank around which a learner at ``difficulty`` stops knowing words."""

    if lower is not None:
        difficulty = min(difficulty, lower)
    span = policy.max_expected_frequency_rank / policy.base_expected_frequency_rank
    return int(round(policy.base_expected_frequency_rank * span**difficulty))
