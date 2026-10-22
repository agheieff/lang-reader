from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Literal

import pytest

from server.calibration import (
    CEFR_BOUNDARIES,
    CEFR_CENTERS,
    CEFR_LEVELS,
    MIN_REPORTED_INTERVAL_WIDTH,
    CalibrationAttempt,
    CalibrationPrior,
    OrdinaryReadingAttempt,
    ProbeObservation,
    cefr_center,
    cefr_for_difficulty,
    estimate_calibration,
    filter_qualified_attempts,
    filter_qualified_reading_attempts,
    ordinary_reading_outcome,
)


def _probes(
    *,
    count: int = 8,
    revealed: bool = False,
    start: float = 0.20,
    step: float = 0.05,
    translated: Iterable[int] = (),
) -> tuple[ProbeObservation, ...]:
    translated_indices = set(translated)
    return tuple(
        ProbeObservation(
            term_key=f"term-{index}",
            difficulty=min(start + step * index, 0.98),
            revealed=revealed,
            sentence_translated=index in translated_indices,
        )
        for index in range(count)
    )


def _attempt(
    attempt_id: str,
    *,
    probes: tuple[ProbeObservation, ...] | None = None,
    active_seconds: float = 60.0,
    completion_ratio: float = 1.0,
    full_translation: bool = False,
    occurred_at: datetime | None = None,
) -> CalibrationAttempt:
    return CalibrationAttempt(
        attempt_id=attempt_id,
        active_seconds=active_seconds,
        completion_ratio=completion_ratio,
        full_translation_revealed=full_translation,
        probes=probes or _probes(),
        occurred_at=occurred_at,
    )


def _reading_attempt(
    attempt_id: str,
    *,
    difficulty: float = 0.30,
    active_seconds: float = 60.0,
    completion_ratio: float = 1.0,
    terms: int = 20,
    reveals: int = 0,
    sentences: int = 5,
    translated_sentences: int = 0,
    full_translation: bool = False,
    feedback: tuple[Literal["easier", "more_challenging"], ...] = (),
    reliability: float = 1.0,
    occurred_at: datetime | None = None,
) -> OrdinaryReadingAttempt:
    return OrdinaryReadingAttempt(
        attempt_id=attempt_id,
        difficulty=difficulty,
        active_seconds=active_seconds,
        completion_ratio=completion_ratio,
        eligible_term_count=terms,
        revealed_term_count=reveals,
        sentence_count=sentences,
        translated_sentence_count=translated_sentences,
        full_translation_revealed=full_translation,
        difficulty_feedback=feedback,
        difficulty_reliability=reliability,
        occurred_at=occurred_at,
    )


def test_cefr_boundaries_and_centers() -> None:
    assert pytest.approx(tuple(index / 6 for index in range(7))) == CEFR_BOUNDARIES
    for index, level in enumerate(CEFR_LEVELS):
        lower = index / 6
        center = (index + 0.5) / 6
        assert cefr_for_difficulty(lower) == level
        assert cefr_for_difficulty(center) == level
        assert cefr_center(level) == pytest.approx(center)
        assert CEFR_CENTERS[level] == pytest.approx(center)
    assert cefr_for_difficulty(1.0) == "C2"


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_difficulties_fail_fast(value: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        cefr_for_difficulty(value)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ProbeObservation(term_key="probe", difficulty=value, revealed=False)


def test_attempt_input_validation() -> None:
    with pytest.raises(ValueError, match="term_key"):
        ProbeObservation(term_key=" ", difficulty=0.5, revealed=False)
    with pytest.raises(ValueError, match="attempt_id"):
        _attempt(" ")
    with pytest.raises(ValueError, match="active_seconds"):
        _attempt("bad-time", active_seconds=-1)
    with pytest.raises(ValueError, match="completion_ratio"):
        _attempt("bad-ratio", completion_ratio=1.1)
    duplicate = ProbeObservation(term_key="duplicate", difficulty=0.5, revealed=False)
    with pytest.raises(ValueError, match="unique"):
        _attempt("duplicates", probes=(duplicate, duplicate))


def test_attempt_qualification_is_conservative() -> None:
    translated = _attempt("translated", probes=_probes(count=10, translated=(0, 1)))
    qualified = filter_qualified_attempts(
        [
            _attempt("short", active_seconds=29.99),
            _attempt("incomplete", completion_ratio=0.79),
            _attempt("full", full_translation=True),
            _attempt("too-few", probes=_probes(count=10, translated=(0, 1, 2))),
            translated,
        ]
    )

    assert [attempt.attempt_id for attempt in qualified] == ["translated"]
    assert len(qualified[0].probes) == 8
    assert all(not probe.sentence_translated for probe in qualified[0].probes)


def test_no_qualified_attempt_has_no_estimate() -> None:
    result = estimate_calibration(
        [_attempt("short", active_seconds=2), _attempt("translated", full_translation=True)]
    )

    assert result.status == "unstarted"
    assert result.difficulty is None
    assert result.interval_width is None
    assert result.qualified_attempts == 0
    assert result.usable_probes == 0


def test_questionnaire_prior_seeds_an_unstarted_replay_without_counting_as_evidence() -> None:
    beginner = estimate_calibration([], prior=CalibrationPrior(mean=0.02, variance=0.08**2))
    near_native = estimate_calibration([], prior=CalibrationPrior(mean=0.94, variance=0.08**2))

    assert beginner.status == near_native.status == "unstarted"
    assert beginner.qualified_attempts == near_native.qualified_attempts == 0
    assert beginner.usable_probes == near_native.usable_probes == 0
    # The declared prior remains the center while its truncated distribution supplies uncertainty.
    assert (beginner.difficulty, beginner.lower, beginner.upper) == (0.02, 0.01, 0.145)
    assert (near_native.difficulty, near_native.lower, near_native.upper) == (
        0.94,
        0.825,
        0.985,
    )


def test_timestamped_calibration_recency_is_deterministic_and_fades_toward_prior() -> None:
    observed_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prior = CalibrationPrior(mean=0.5, variance=0.04)
    known = _attempt(
        "known",
        probes=_probes(revealed=False, start=0.30, step=0.04),
        occurred_at=observed_at,
    )

    fresh = estimate_calibration([known], prior=prior, as_of=observed_at)
    one_year_old = estimate_calibration(
        [known],
        prior=prior,
        as_of=observed_at + timedelta(days=365),
    )
    two_years_old = estimate_calibration(
        [known],
        prior=prior,
        as_of=observed_at + timedelta(days=730),
    )

    assert fresh.difficulty == 0.70
    assert one_year_old.difficulty == 0.645
    assert two_years_old.difficulty == 0.59
    assert fresh.difficulty > one_year_old.difficulty > two_years_old.difficulty > prior.mean
    assert fresh.interval_width is not None
    assert one_year_old.interval_width is not None
    assert two_years_old.interval_width is not None
    assert fresh.interval_width < one_year_old.interval_width < two_years_old.interval_width


def test_newer_contradictory_attempt_dominates_independently_of_input_order() -> None:
    old_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    new_at = old_at + timedelta(days=365)
    prior = CalibrationPrior(mean=0.5, variance=0.04)
    old_known = _attempt(
        "old-known",
        probes=_probes(revealed=False, start=0.30, step=0.04),
        occurred_at=old_at,
    )
    new_revealed = _attempt(
        "new-revealed",
        probes=_probes(revealed=True, start=0.30, step=0.04),
        occurred_at=new_at,
    )
    old_revealed = _attempt(
        "old-revealed",
        probes=_probes(revealed=True, start=0.30, step=0.04),
        occurred_at=old_at,
    )
    new_known = _attempt(
        "new-known",
        probes=_probes(revealed=False, start=0.30, step=0.04),
        occurred_at=new_at,
    )

    recent_failure = estimate_calibration([old_known, new_revealed], prior=prior, as_of=new_at)
    recent_success = estimate_calibration([old_revealed, new_known], prior=prior, as_of=new_at)

    assert recent_failure.difficulty == 0.245
    assert recent_success.difficulty == 0.445
    assert recent_failure.difficulty < recent_success.difficulty
    assert (
        estimate_calibration([new_revealed, old_known], prior=prior, as_of=new_at) == recent_failure
    )


def test_reveals_move_estimate_down_and_non_reveals_move_it_up() -> None:
    revealed = estimate_calibration([_attempt("revealed", probes=_probes(revealed=True))])
    unrevealed = estimate_calibration([_attempt("known", probes=_probes(revealed=False))])

    assert revealed.status == "collecting"
    assert unrevealed.status == "collecting"
    assert revealed.difficulty is not None
    assert unrevealed.difficulty is not None
    assert revealed.difficulty < 0.30
    assert unrevealed.difficulty > 0.60
    assert revealed.difficulty < unrevealed.difficulty


def test_two_attempts_produce_a_rough_estimate() -> None:
    result = estimate_calibration(
        [_attempt("one", probes=_probes(count=12)), _attempt("two", probes=_probes(count=12))]
    )

    assert result.status == "rough"
    assert result.qualified_attempts == 2
    assert result.usable_probes == 24
    assert result.difficulty is not None
    assert result.lower is not None
    assert result.upper is not None
    assert result.lower <= result.difficulty <= result.upper
    assert result.level is not None
    assert result.lower_level is not None
    assert result.upper_level is not None


def test_four_consistent_attempts_can_become_stable() -> None:
    attempts: list[CalibrationAttempt] = []
    for attempt_index in range(4):
        probes = tuple(
            ProbeObservation(
                term_key=f"term-{probe_index}",
                difficulty=0.20 + probe_index * 0.03,
                revealed=0.20 + probe_index * 0.03 > 0.50,
            )
            for probe_index in range(20)
        )
        attempts.append(_attempt(f"attempt-{attempt_index}", probes=probes))

    result = estimate_calibration(attempts)

    assert result.status == "stable"
    assert result.interval_width is not None
    assert MIN_REPORTED_INTERVAL_WIDTH <= result.interval_width <= 0.18


def test_estimation_is_order_independent_and_attempt_idempotent() -> None:
    first = _attempt("first", probes=_probes(count=12, revealed=False))
    second = _attempt("second", probes=_probes(count=12, revealed=True))
    reordered_first = _attempt("first", probes=tuple(reversed(first.probes)))

    expected = estimate_calibration([first, second])
    assert estimate_calibration([second, first]) == expected
    assert estimate_calibration([first, second, reordered_first]) == expected


def test_conflicting_attempt_reuse_fails_fast() -> None:
    first = _attempt("same", active_seconds=60)
    conflicting = _attempt("same", active_seconds=61)

    with pytest.raises(ValueError, match="attempt_id reused"):
        estimate_calibration([first, conflicting])


def test_ordinary_reading_qualification_is_deduplicated_and_conservative() -> None:
    valid = _reading_attempt("valid")
    qualified = filter_qualified_reading_attempts(
        [
            _reading_attempt("short", active_seconds=29.99),
            _reading_attempt("partial", completion_ratio=0.79),
            valid,
            valid,
        ]
    )

    assert qualified == (valid,)
    with pytest.raises(ValueError, match="attempt_id reused"):
        filter_qualified_reading_attempts([valid, _reading_attempt("valid", reveals=1)])


def test_ordinary_word_reveals_are_weaker_than_direct_translation_help() -> None:
    clean = _reading_attempt("clean")
    checks = _reading_attempt("checks", reveals=5)
    sentence_help = _reading_attempt("sentence-help", translated_sentences=2)
    full_help = _reading_attempt("full-help", full_translation=True)

    assert ordinary_reading_outcome(clean) == pytest.approx(0.80)
    assert ordinary_reading_outcome(checks) > 0.75
    assert ordinary_reading_outcome(checks) > ordinary_reading_outcome(sentence_help)
    assert ordinary_reading_outcome(sentence_help) > ordinary_reading_outcome(full_help)


def test_ordinary_success_moves_broad_estimate_quickly_then_slows() -> None:
    prior = CalibrationPrior(mean=0.30, variance=0.18**2)
    attempts = [_reading_attempt(f"reading-{index}") for index in range(8)]

    baseline = estimate_calibration([], prior=prior)
    first = estimate_calibration([], prior=prior, reading_attempts=attempts[:1])
    second = estimate_calibration([], prior=prior, reading_attempts=attempts[:2])
    fourth = estimate_calibration([], prior=prior, reading_attempts=attempts[:4])
    eighth = estimate_calibration([], prior=prior, reading_attempts=attempts)

    assert baseline.difficulty is not None
    assert first.difficulty is not None
    assert second.difficulty is not None
    assert fourth.difficulty is not None
    assert eighth.difficulty is not None
    assert baseline.difficulty < first.difficulty < second.difficulty < fourth.difficulty
    assert fourth.difficulty < eighth.difficulty
    assert first.difficulty - baseline.difficulty > eighth.difficulty - fourth.difficulty
    assert eighth.status == "rough"
    assert eighth.qualified_readings == 8
    assert eighth.qualified_attempts == eighth.usable_probes == 0


def test_explicit_difficulty_feedback_outweighs_ordinary_word_checks() -> None:
    prior = CalibrationPrior(mean=0.30, variance=0.18**2)
    checks = estimate_calibration(
        [],
        prior=prior,
        reading_attempts=[_reading_attempt("checks", reveals=5)],
    )
    easier = estimate_calibration(
        [],
        prior=prior,
        reading_attempts=[_reading_attempt("easier", feedback=("easier",))],
    )
    harder = estimate_calibration(
        [],
        prior=prior,
        reading_attempts=[_reading_attempt("harder", feedback=("more_challenging",))],
    )

    assert easier.difficulty is not None
    assert checks.difficulty is not None
    assert harder.difficulty is not None
    assert easier.difficulty < checks.difficulty < harder.difficulty


def test_legacy_lesson_difficulty_remains_useful_but_is_conservatively_weighted() -> None:
    prior = CalibrationPrior(mean=0.30, variance=0.18**2)
    legacy = estimate_calibration(
        [],
        prior=prior,
        reading_attempts=[_reading_attempt("legacy", reliability=0.5)],
    )
    assigned = estimate_calibration(
        [],
        prior=prior,
        reading_attempts=[_reading_attempt("assigned")],
    )
    baseline = estimate_calibration([], prior=prior)

    assert baseline.difficulty is not None
    assert legacy.difficulty is not None
    assert assigned.difficulty is not None
    assert baseline.difficulty < legacy.difficulty < assigned.difficulty


def test_recent_ordinary_feedback_dominates_old_conflicting_feedback() -> None:
    old_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    new_at = old_at + timedelta(days=365)
    prior = CalibrationPrior(mean=0.50, variance=0.18**2)
    old_harder = _reading_attempt(
        "old-harder",
        difficulty=0.50,
        feedback=("more_challenging",),
        occurred_at=old_at,
    )
    fresh_easier = _reading_attempt(
        "fresh-easier",
        difficulty=0.50,
        feedback=("easier",),
        occurred_at=new_at,
    )
    old_easier = _reading_attempt(
        "old-easier",
        difficulty=0.50,
        feedback=("easier",),
        occurred_at=old_at,
    )
    fresh_harder = _reading_attempt(
        "fresh-harder",
        difficulty=0.50,
        feedback=("more_challenging",),
        occurred_at=new_at,
    )

    lower = estimate_calibration(
        [],
        prior=prior,
        reading_attempts=[old_harder, fresh_easier],
        as_of=new_at,
    )
    higher = estimate_calibration(
        [],
        prior=prior,
        reading_attempts=[old_easier, fresh_harder],
        as_of=new_at,
    )

    assert lower.difficulty is not None
    assert higher.difficulty is not None
    assert lower.difficulty < higher.difficulty
