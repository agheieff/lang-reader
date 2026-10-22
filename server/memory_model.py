"""FSRS-4.5 memory model adapted to implicit reading evidence.

A reading app never asks for a grade. A reveal is an explicit lapse ("again"); finishing a qualified
reading without revealing a word is weaker evidence of recall, applied as a partial "good" review
whose weight is ``passive_confidence``. FSRS makes massed exposure nearly worthless by itself: when
predicted recall is already close to 1, a successful review barely raises stability.

``knowledge`` is the probability of still recalling a word ``horizon_days`` after now. It decays
without evidence and only grows with spaced success, which is what the learning bands mean. Words
without evidence fall back to a frequency prior relative to the learner's reading frontier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

SECONDS_PER_DAY = 86_400.0
DECAY = -0.5
FACTOR = 19.0 / 81.0
AGAIN = 1
GOOD = 3

# FSRS-4.5 default parameters (open-spaced-repetition), fitted on a large review corpus.
FSRS_DEFAULT_WEIGHTS: tuple[float, ...] = (
    0.4872,
    1.4003,
    3.7145,
    13.8206,
    5.1618,
    1.2298,
    0.8975,
    0.031,
    1.6474,
    0.1367,
    1.0461,
    2.1072,
    0.0793,
    0.3246,
    1.587,
    0.2272,
    2.8755,
)


@dataclass(frozen=True)
class MemoryPolicy:
    weights: tuple[float, ...] = FSRS_DEFAULT_WEIGHTS
    # How strongly an unrevealed word in a qualified reading counts as a successful review.
    passive_confidence: float = 0.6
    # Further discount for passive evidence when the same lesson was already read before.
    reread_weight: float = 0.3
    horizon_days: float = 30.0
    minimum_stability_days: float = 0.05
    maximum_stability_days: float = 3650.0
    # Logistic slope of P(known) over log frequency rank around the learner's frontier rank.
    prior_slope: float = 1.2
    prior_ceiling: float = 0.97
    prior_floor: float = 0.03

    def __post_init__(self) -> None:
        if len(self.weights) != 17 or any(not math.isfinite(w) for w in self.weights):
            raise ValueError("FSRS-4.5 needs exactly 17 finite weights")
        for name in ("passive_confidence", "reread_weight"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if any(
            not math.isfinite(value)
            for value in (
                self.horizon_days,
                self.prior_slope,
                self.minimum_stability_days,
                self.maximum_stability_days,
            )
        ):
            raise ValueError("memory policy values must be finite")
        if not 0 < self.minimum_stability_days < self.maximum_stability_days:
            raise ValueError("stability bounds must be positive and ordered")
        if self.horizon_days < 0 or self.prior_slope <= 0:
            raise ValueError("horizon must be non-negative and prior slope positive")
        if not 0.0 < self.prior_floor < self.prior_ceiling < 1.0:
            raise ValueError("prior bounds must be ordered within (0, 1)")


DEFAULT_MEMORY_POLICY = MemoryPolicy()


@dataclass(frozen=True)
class MemoryState:
    difficulty: float
    stability_days: float
    last_review_at: datetime


def elapsed_days(since: datetime, at: datetime) -> float:
    return max(0.0, (at - since).total_seconds() / SECONDS_PER_DAY)


def retrievability(stability_days: float, days: float) -> float:
    return float((1.0 + FACTOR * max(0.0, days) / stability_days) ** DECAY)


def knowledge(
    state: MemoryState | None,
    *,
    prior: float,
    at: datetime,
    policy: MemoryPolicy = DEFAULT_MEMORY_POLICY,
) -> float:
    """Probability of recall ``horizon_days`` from ``at``; the prior when there is no evidence."""

    if state is None:
        return prior
    days = elapsed_days(state.last_review_at, at) + policy.horizon_days
    return retrievability(state.stability_days, days)


def recall_now(
    state: MemoryState | None,
    *,
    prior: float,
    at: datetime,
) -> float:
    if state is None:
        return prior
    return retrievability(state.stability_days, elapsed_days(state.last_review_at, at))


def prior_known(
    frequency_rank: int | None,
    frontier_rank: float,
    policy: MemoryPolicy = DEFAULT_MEMORY_POLICY,
) -> float:
    """P(word already known) from corpus frequency relative to the learner's frontier rank."""

    if frequency_rank is None:
        return 0.5
    z = policy.prior_slope * (math.log(frontier_rank) - math.log(max(1, frequency_rank)))
    value = 1.0 / (1.0 + math.exp(-z))
    return min(policy.prior_ceiling, max(policy.prior_floor, value))


def prior_state(prior: float, at: datetime, policy: MemoryPolicy) -> MemoryState:
    """The memory state whose recall at the horizon equals ``prior`` (pre-existing knowledge)."""

    horizon = max(1.0, policy.horizon_days)
    stability = FACTOR * horizon / (prior ** (1 / DECAY) - 1)
    return MemoryState(
        difficulty=_clamp_difficulty(policy.weights[4]),
        stability_days=_bounded_stability(stability, policy),
        last_review_at=at,
    )


def review(
    state: MemoryState | None,
    grade: int,
    at: datetime,
    *,
    weight: float = 1.0,
    prior: float | None = None,
    policy: MemoryPolicy = DEFAULT_MEMORY_POLICY,
) -> MemoryState:
    """Apply one (possibly partial) review; ``weight`` interpolates toward the full outcome.

    For the first evidence, ``prior`` seeds knowledge the learner may already have from outside
    the app: the result is the more stable of a fresh FSRS start and the same review applied to
    the prior-implied state. A lapse on a common word therefore lowers, but does not erase, what
    the prior implied; a rare word starts from FSRS's own initial values.
    """

    if state is None and prior is not None:
        fresh = review(None, grade, at, weight=weight, policy=policy)
        seeded = review(prior_state(prior, at, policy), grade, at, weight=weight, policy=policy)
        return fresh if fresh.stability_days >= seeded.stability_days else seeded
    if grade not in (AGAIN, GOOD):
        raise ValueError("reading evidence only produces again/good grades")
    if not 0.0 < weight <= 1.0:
        raise ValueError("review weight must be in (0, 1]")
    w = policy.weights
    if state is None:
        full_stability = w[grade - 1]
        full_difficulty = _clamp_difficulty(w[4] - (grade - 3) * w[5])
        if weight < 1.0:
            # Partial first evidence starts between a first lapse and a first full success.
            base_stability = w[AGAIN - 1]
            base_difficulty = _clamp_difficulty(w[4] - (AGAIN - 3) * w[5])
            full_stability = base_stability + weight * (full_stability - base_stability)
            full_difficulty = base_difficulty + weight * (full_difficulty - base_difficulty)
        return MemoryState(
            difficulty=full_difficulty,
            stability_days=_bounded_stability(full_stability, policy),
            last_review_at=at,
        )

    difficulty = state.difficulty
    stability = state.stability_days
    recall = retrievability(stability, elapsed_days(state.last_review_at, at))
    next_difficulty = _clamp_difficulty(
        w[7] * _clamp_difficulty(w[4]) + (1 - w[7]) * (difficulty - w[6] * (grade - 3))
    )
    if grade == AGAIN:
        next_stability = min(
            stability,
            w[11]
            * difficulty ** (-w[12])
            * ((stability + 1) ** w[13] - 1)
            * math.exp(w[14] * (1 - recall)),
        )
    else:
        next_stability = stability * (
            1
            + math.exp(w[8])
            * (11 - difficulty)
            * stability ** (-w[9])
            * (math.exp(w[10] * (1 - recall)) - 1)
        )
    return MemoryState(
        difficulty=difficulty + weight * (next_difficulty - difficulty),
        stability_days=_bounded_stability(
            stability + weight * (next_stability - stability), policy
        ),
        last_review_at=at,
    )


def interval_days(stability_days: float, desired_retention: float = 0.9) -> float:
    """Days until predicted recall falls to ``desired_retention``."""

    if not math.isfinite(stability_days) or stability_days <= 0:
        raise ValueError("stability must be finite and positive")
    if not 0 < desired_retention <= 1:
        raise ValueError("desired retention must be in (0, 1]")
    return float(stability_days / FACTOR * (desired_retention ** (1 / DECAY) - 1))


def _clamp_difficulty(value: float) -> float:
    return min(10.0, max(1.0, value))


def _bounded_stability(value: float, policy: MemoryPolicy) -> float:
    return min(policy.maximum_stability_days, max(policy.minimum_stability_days, value))
