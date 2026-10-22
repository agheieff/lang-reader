"""Pure, deterministic reading-level calibration.

The persistence layer turns completed lesson sessions into these small value objects. This module
deliberately knows nothing about SQLAlchemy, API payloads, or lesson schemas.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from server.reading_evidence import is_qualified_reading

CefrLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
CalibrationStatus = Literal["unstarted", "collecting", "rough", "stable"]

CEFR_LEVELS: tuple[CefrLevel, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")
CEFR_BOUNDARIES: tuple[float, ...] = tuple(index / 6 for index in range(7))
CEFR_CENTERS: Mapping[CefrLevel, float] = {
    level: (index + 0.5) / 6 for index, level in enumerate(CEFR_LEVELS)
}
INITIAL_PROBE_DIFFICULTY = CEFR_CENTERS["B1"]

GRID_STEPS = 200
GRID: tuple[float, ...] = tuple(index / GRID_STEPS for index in range(GRID_STEPS + 1))
LOGISTIC_SCALE = 0.10
REVEAL_PROBABILITY_IF_KNOWN = 0.05
REVEAL_PROBABILITY_IF_UNKNOWN = 0.65
MAX_EFFECTIVE_PROBES_PER_ATTEMPT = 8.0
EVIDENCE_HALF_LIFE_DAYS = 365.0
ORDINARY_READING_EVIDENCE_WEIGHT = 0.75
EXPLICIT_DIFFICULTY_FEEDBACK_WEIGHT = 2.0
READING_COMFORT_SCALE = 0.10
BASE_COMPLETION_OUTCOME = 0.80
MAX_TERM_REVEAL_PENALTY = 0.25
MAX_SENTENCE_TRANSLATION_PENALTY = 0.45
FULL_TRANSLATION_MAX_OUTCOME = 0.20
EASIER_FEEDBACK_MAX_OUTCOME = 0.10
MORE_CHALLENGING_FEEDBACK_MIN_OUTCOME = 0.95

MIN_USABLE_PROBES_PER_ATTEMPT = 8
ROUGH_MIN_ATTEMPTS = 2
ROUGH_MIN_PROBES = 12
STABLE_MIN_ATTEMPTS = 4
# Ordinary first readings are weaker, coarser evidence than calibration probes.
ROUGH_MIN_READINGS = 5
STABLE_MIN_READINGS = 12
STABLE_MAX_INTERVAL_WIDTH = 0.18
MIN_REPORTED_INTERVAL_WIDTH = 0.12
LOWER_QUANTILE = 0.10
UPPER_QUANTILE = 0.90


def cefr_for_difficulty(difficulty: float) -> CefrLevel:
    """Map a continuous difficulty to its containing equal-width CEFR band."""

    _validate_unit_interval(difficulty, "difficulty")
    index = min(int(difficulty * len(CEFR_LEVELS)), len(CEFR_LEVELS) - 1)
    return CEFR_LEVELS[index]


def cefr_center(level: CefrLevel) -> float:
    return CEFR_CENTERS[level]


@dataclass(frozen=True)
class ProbeObservation:
    """Whether a learner revealed one authored calibration probe."""

    term_key: str
    difficulty: float
    revealed: bool
    sentence_translated: bool = False

    def __post_init__(self) -> None:
        if not self.term_key.strip():
            raise ValueError("term_key must not be blank")
        _validate_unit_interval(self.difficulty, "probe difficulty")


@dataclass(frozen=True)
class CalibrationAttempt:
    """All calibration inputs from one completed lesson session."""

    attempt_id: str
    active_seconds: float
    completion_ratio: float
    full_translation_revealed: bool
    probes: tuple[ProbeObservation, ...]
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id.strip():
            raise ValueError("attempt_id must not be blank")
        if not math.isfinite(self.active_seconds) or self.active_seconds < 0:
            raise ValueError("active_seconds must be finite and non-negative")
        _validate_unit_interval(self.completion_ratio, "completion_ratio")
        keys = [probe.term_key for probe in self.probes]
        if len(keys) != len(set(keys)):
            raise ValueError("probe term keys must be unique within an attempt")
        if self.occurred_at is not None and self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")


@dataclass(frozen=True)
class CalibrationPrior:
    """A weak questionnaire seed; it is not counted as a calibration attempt."""

    mean: float
    variance: float

    def __post_init__(self) -> None:
        _validate_unit_interval(self.mean, "prior mean")
        if not math.isfinite(self.variance) or self.variance <= 0:
            raise ValueError("prior variance must be finite and positive")


@dataclass(frozen=True)
class OrdinaryReadingAttempt:
    """A deliberately coarse challenge signal from one lesson's first completed reading.

    Ordinary lessons are not calibrated tests. Their evidence is therefore session-level and
    weaker than authored probes: completing a text is positive, direct translation help is
    negative, and individual word reveals only make a small nonlinear adjustment because a click
    can be a meaning or pronunciation check.
    """

    attempt_id: str
    difficulty: float
    active_seconds: float
    completion_ratio: float
    eligible_term_count: int
    revealed_term_count: int
    sentence_count: int
    translated_sentence_count: int
    full_translation_revealed: bool = False
    difficulty_feedback: tuple[Literal["easier", "more_challenging"], ...] = ()
    difficulty_reliability: float = 1.0
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id.strip():
            raise ValueError("attempt_id must not be blank")
        _validate_unit_interval(self.difficulty, "lesson difficulty")
        if not math.isfinite(self.active_seconds) or self.active_seconds < 0:
            raise ValueError("active_seconds must be finite and non-negative")
        _validate_unit_interval(self.completion_ratio, "completion_ratio")
        if self.eligible_term_count < 0:
            raise ValueError("eligible_term_count must be non-negative")
        if not 0 <= self.revealed_term_count <= self.eligible_term_count:
            raise ValueError("revealed_term_count must be within eligible_term_count")
        if self.sentence_count < 1:
            raise ValueError("sentence_count must be positive")
        if not 0 <= self.translated_sentence_count <= self.sentence_count:
            raise ValueError("translated_sentence_count must be within sentence_count")
        if len(self.difficulty_feedback) != len(set(self.difficulty_feedback)):
            raise ValueError("difficulty_feedback must be unique")
        if not math.isfinite(self.difficulty_reliability) or not (
            0.0 < self.difficulty_reliability <= 1.0
        ):
            raise ValueError("difficulty_reliability must be finite and between 0 and 1")
        if self.occurred_at is not None and self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")


@dataclass(frozen=True)
class QualifiedAttempt:
    attempt_id: str
    probes: tuple[ProbeObservation, ...]
    occurred_at: datetime | None = None


@dataclass(frozen=True)
class CalibrationEstimate:
    status: CalibrationStatus
    difficulty: float | None
    lower: float | None
    upper: float | None
    level: CefrLevel | None
    lower_level: CefrLevel | None
    upper_level: CefrLevel | None
    qualified_attempts: int
    usable_probes: int
    qualified_readings: int = 0

    @property
    def interval_width(self) -> float | None:
        if self.lower is None or self.upper is None:
            return None
        return self.upper - self.lower


def filter_qualified_attempts(
    attempts: Iterable[CalibrationAttempt],
) -> tuple[QualifiedAttempt, ...]:
    """Deduplicate, validate, and retain attempts suitable for calibration."""

    unique: dict[str, CalibrationAttempt] = {}
    fingerprints: dict[str, tuple[object, ...]] = {}
    for attempt in attempts:
        fingerprint = _attempt_fingerprint(attempt)
        previous = fingerprints.get(attempt.attempt_id)
        if previous is not None:
            if previous != fingerprint:
                raise ValueError(f"attempt_id reused with different data: {attempt.attempt_id}")
            continue
        unique[attempt.attempt_id] = attempt
        fingerprints[attempt.attempt_id] = fingerprint

    result: list[QualifiedAttempt] = []
    for attempt_id in sorted(unique):
        attempt = unique[attempt_id]
        if attempt.full_translation_revealed or not is_qualified_reading(
            active_seconds=attempt.active_seconds,
            completion_ratio=attempt.completion_ratio,
        ):
            continue
        probes = tuple(
            sorted(
                (probe for probe in attempt.probes if not probe.sentence_translated),
                key=_probe_sort_key,
            )
        )
        if len(probes) < MIN_USABLE_PROBES_PER_ATTEMPT:
            continue
        result.append(
            QualifiedAttempt(
                attempt_id=attempt_id,
                probes=probes,
                occurred_at=attempt.occurred_at,
            )
        )
    return tuple(result)


def filter_qualified_reading_attempts(
    attempts: Iterable[OrdinaryReadingAttempt],
) -> tuple[OrdinaryReadingAttempt, ...]:
    """Deduplicate and retain first-reading inputs that contain meaningful active time."""

    unique: dict[str, OrdinaryReadingAttempt] = {}
    fingerprints: dict[str, tuple[object, ...]] = {}
    for attempt in attempts:
        fingerprint = _reading_attempt_fingerprint(attempt)
        previous = fingerprints.get(attempt.attempt_id)
        if previous is not None:
            if previous != fingerprint:
                raise ValueError(f"attempt_id reused with different data: {attempt.attempt_id}")
            continue
        unique[attempt.attempt_id] = attempt
        fingerprints[attempt.attempt_id] = fingerprint
    ordered = (unique[attempt_id] for attempt_id in sorted(unique))
    return tuple(
        attempt
        for attempt in ordered
        if is_qualified_reading(
            active_seconds=attempt.active_seconds,
            completion_ratio=attempt.completion_ratio,
        )
    )


def ordinary_reading_outcome(attempt: OrdinaryReadingAttempt) -> float:
    """Return a fractional easy-reading outcome while keeping word clicks weak evidence."""

    term_rate = (
        attempt.revealed_term_count / attempt.eligible_term_count
        if attempt.eligible_term_count
        else 0.0
    )
    sentence_rate = attempt.translated_sentence_count / attempt.sentence_count
    outcome: float = float(
        BASE_COMPLETION_OUTCOME
        - MAX_TERM_REVEAL_PENALTY * term_rate**1.5
        - MAX_SENTENCE_TRANSLATION_PENALTY * sentence_rate**0.7
    )
    if attempt.full_translation_revealed:
        outcome = min(outcome, FULL_TRANSLATION_MAX_OUTCOME)
    feedback = set(attempt.difficulty_feedback)
    if feedback == {"easier"}:
        outcome = min(outcome, EASIER_FEEDBACK_MAX_OUTCOME)
    elif feedback == {"more_challenging"}:
        outcome = max(outcome, MORE_CHALLENGING_FEEDBACK_MIN_OUTCOME)
    return max(0.05, min(0.95, outcome))


def estimate_calibration(
    attempts: Iterable[CalibrationAttempt],
    *,
    prior: CalibrationPrior | None = None,
    as_of: datetime | None = None,
    reading_attempts: Iterable[OrdinaryReadingAttempt] = (),
) -> CalibrationEstimate:
    """Estimate reading ability with a replayable, confidence-sensitive Bayesian/Elo update.

    A broad posterior moves quickly; accumulated observations narrow it and make later movement
    progressively smaller. Timestamped evidence fades gradually so a stale estimate can adapt
    again. Untimestamped inputs retain deterministic equal weighting for pure callers and tests.
    """

    qualified = filter_qualified_attempts(attempts)
    qualified_readings = filter_qualified_reading_attempts(reading_attempts)
    if not qualified and prior is None:
        return CalibrationEstimate(
            status="unstarted",
            difficulty=None,
            lower=None,
            upper=None,
            level=None,
            lower_level=None,
            upper_level=None,
            qualified_attempts=0,
            usable_probes=0,
        )

    log_weights = _prior_log_weights(prior)
    reference_at = _reference_time(qualified, qualified_readings, as_of)
    for attempt in qualified:
        observation_weight = min(
            1.0, MAX_EFFECTIVE_PROBES_PER_ATTEMPT / len(attempt.probes)
        ) * _recency_weight(attempt.occurred_at, reference_at)
        for probe in attempt.probes:
            for index, ability in enumerate(GRID):
                reveal_probability = _reveal_probability(ability, probe.difficulty)
                likelihood = reveal_probability if probe.revealed else 1.0 - reveal_probability
                log_weights[index] += observation_weight * math.log(likelihood)
    for reading_attempt in qualified_readings:
        outcome = ordinary_reading_outcome(reading_attempt)
        explicit_feedback = len(reading_attempt.difficulty_feedback) == 1
        observation_weight = (
            ORDINARY_READING_EVIDENCE_WEIGHT
            * reading_attempt.difficulty_reliability
            * _recency_weight(reading_attempt.occurred_at, reference_at)
            * (EXPLICIT_DIFFICULTY_FEEDBACK_WEIGHT if explicit_feedback else 1.0)
        )
        for index, ability in enumerate(GRID):
            comfort_probability = _reading_comfort_probability(ability, reading_attempt.difficulty)
            log_weights[index] += observation_weight * (
                outcome * math.log(comfort_probability)
                + (1.0 - outcome) * math.log(1.0 - comfort_probability)
            )

    probabilities = _normalize_log_weights(log_weights)
    difficulty = (
        prior.mean
        if prior is not None and not qualified and not qualified_readings
        else _quantile(probabilities, 0.5)
    )
    lower = _quantile(probabilities, LOWER_QUANTILE)
    upper = _quantile(probabilities, UPPER_QUANTILE)
    lower, upper = _apply_interval_floor(lower, upper, difficulty)
    usable_probes = sum(len(attempt.probes) for attempt in qualified)
    status = _status(len(qualified), usable_probes, upper - lower, len(qualified_readings))

    return CalibrationEstimate(
        status=status if qualified or qualified_readings else "unstarted",
        difficulty=difficulty,
        lower=lower,
        upper=upper,
        level=cefr_for_difficulty(difficulty),
        lower_level=cefr_for_difficulty(lower),
        upper_level=cefr_for_difficulty(upper),
        qualified_attempts=len(qualified),
        usable_probes=usable_probes,
        qualified_readings=len(qualified_readings),
    )


def _status(
    attempts: int, probes: int, interval_width: float, readings: int = 0
) -> CalibrationStatus:
    if interval_width <= STABLE_MAX_INTERVAL_WIDTH and (
        attempts >= STABLE_MIN_ATTEMPTS or readings >= STABLE_MIN_READINGS
    ):
        return "stable"
    if (attempts >= ROUGH_MIN_ATTEMPTS and probes >= ROUGH_MIN_PROBES) or (
        readings >= ROUGH_MIN_READINGS
    ):
        return "rough"
    return "collecting"


def _reveal_probability(ability: float, probe_difficulty: float) -> float:
    known_probability = 1.0 / (1.0 + math.exp(-(ability - probe_difficulty) / LOGISTIC_SCALE))
    return REVEAL_PROBABILITY_IF_KNOWN * known_probability + REVEAL_PROBABILITY_IF_UNKNOWN * (
        1.0 - known_probability
    )


def _normalize_log_weights(log_weights: Sequence[float]) -> tuple[float, ...]:
    maximum = max(log_weights)
    weights = tuple(math.exp(value - maximum) for value in log_weights)
    total = sum(weights)
    return tuple(value / total for value in weights)


def _quantile(probabilities: Sequence[float], quantile: float) -> float:
    cumulative = 0.0
    for value, probability in zip(GRID, probabilities, strict=True):
        cumulative += probability
        if cumulative >= quantile:
            return value
    return GRID[-1]


def _apply_interval_floor(lower: float, upper: float, center: float) -> tuple[float, float]:
    if upper - lower >= MIN_REPORTED_INTERVAL_WIDTH:
        return lower, upper
    half_width = MIN_REPORTED_INTERVAL_WIDTH / 2
    adjusted_lower = max(0.0, center - half_width)
    adjusted_upper = min(1.0, center + half_width)
    if adjusted_lower == 0.0:
        adjusted_upper = MIN_REPORTED_INTERVAL_WIDTH
    elif adjusted_upper == 1.0:
        adjusted_lower = 1.0 - MIN_REPORTED_INTERVAL_WIDTH
    return adjusted_lower, adjusted_upper


def _attempt_fingerprint(attempt: CalibrationAttempt) -> tuple[object, ...]:
    probes = tuple(sorted(_probe_sort_key(probe) for probe in attempt.probes))
    return (
        attempt.active_seconds,
        attempt.completion_ratio,
        attempt.full_translation_revealed,
        attempt.occurred_at,
        probes,
    )


def _reading_attempt_fingerprint(attempt: OrdinaryReadingAttempt) -> tuple[object, ...]:
    return (
        attempt.difficulty,
        attempt.active_seconds,
        attempt.completion_ratio,
        attempt.eligible_term_count,
        attempt.revealed_term_count,
        attempt.sentence_count,
        attempt.translated_sentence_count,
        attempt.full_translation_revealed,
        tuple(sorted(attempt.difficulty_feedback)),
        attempt.difficulty_reliability,
        attempt.occurred_at,
    )


def _prior_log_weights(prior: CalibrationPrior | None) -> list[float]:
    if prior is None:
        return [0.0] * len(GRID)
    return [-((ability - prior.mean) ** 2) / (2 * prior.variance) for ability in GRID]


def _reference_time(
    attempts: Sequence[QualifiedAttempt],
    reading_attempts: Sequence[OrdinaryReadingAttempt],
    as_of: datetime | None,
) -> datetime | None:
    timestamps = [attempt.occurred_at for attempt in attempts if attempt.occurred_at is not None]
    timestamps.extend(
        attempt.occurred_at for attempt in reading_attempts if attempt.occurred_at is not None
    )
    if as_of is not None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must include a timezone")
        timestamps.append(as_of)
    return max(timestamps) if timestamps else None


def _recency_weight(occurred_at: datetime | None, reference_at: datetime | None) -> float:
    if occurred_at is None or reference_at is None:
        return 1.0
    age_days = max(0.0, (reference_at - occurred_at).total_seconds() / 86_400)
    return float(0.5 ** (age_days / EVIDENCE_HALF_LIFE_DAYS))


def _reading_comfort_probability(ability: float, difficulty: float) -> float:
    return 1.0 / (1.0 + math.exp(-(ability - difficulty) / READING_COMFORT_SCALE))


def _probe_sort_key(probe: ProbeObservation) -> tuple[str, float, bool, bool]:
    return (
        probe.term_key,
        probe.difficulty,
        probe.revealed,
        probe.sentence_translated,
    )


def _validate_unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
