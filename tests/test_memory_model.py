from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from server.memory_evaluation import Score
from server.memory_model import (
    AGAIN,
    FSRS_DEFAULT_WEIGHTS,
    GOOD,
    MemoryPolicy,
    interval_days,
    knowledge,
    prior_known,
    retrievability,
    review,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_unavailable_evaluation_metrics_are_valid_json_nulls() -> None:
    score = Score(0, 0.0, float("nan"), float("nan"), float("nan")).as_dict()
    assert json.loads(json.dumps(score, allow_nan=False)) == {
        "observations": 0,
        "reveal_rate": 0.0,
        "log_loss": None,
        "brier": None,
        "auc": None,
    }


def test_stability_is_the_time_to_ninety_percent_recall() -> None:
    assert retrievability(7.0, 7.0) == pytest.approx(0.9)
    assert interval_days(7.0) == pytest.approx(7.0)
    assert retrievability(7.0, 0.0) == 1.0


def test_first_reviews_use_fsrs_initial_stability() -> None:
    assert review(None, AGAIN, T0).stability_days == pytest.approx(FSRS_DEFAULT_WEIGHTS[0])
    assert review(None, GOOD, T0).stability_days == pytest.approx(FSRS_DEFAULT_WEIGHTS[2])
    partial = review(None, GOOD, T0, weight=0.5).stability_days
    assert FSRS_DEFAULT_WEIGHTS[0] < partial < FSRS_DEFAULT_WEIGHTS[2]


def test_massed_success_adds_almost_nothing_but_spaced_success_grows_stability() -> None:
    state = review(None, GOOD, T0)
    massed = review(state, GOOD, T0 + timedelta(minutes=5))
    spaced = review(state, GOOD, T0 + timedelta(days=4))

    assert massed.stability_days == pytest.approx(state.stability_days, rel=0.01)
    assert spaced.stability_days > 2 * state.stability_days


def test_a_lapse_never_increases_stability() -> None:
    state = review(review(None, GOOD, T0), GOOD, T0 + timedelta(days=4))
    lapsed = review(state, AGAIN, T0 + timedelta(days=20))
    assert lapsed.stability_days < state.stability_days
    assert lapsed.difficulty > state.difficulty


def test_knowledge_projects_recall_over_the_horizon_and_falls_back_to_the_prior() -> None:
    state = review(None, GOOD, T0)
    assert knowledge(None, prior=0.3, at=T0) == 0.3
    assert knowledge(state, prior=0.3, at=T0) == pytest.approx(
        retrievability(state.stability_days, 30.0)
    )
    assert knowledge(state, prior=0.3, at=T0 + timedelta(days=10)) < knowledge(
        state, prior=0.3, at=T0
    )


def test_frequency_prior_is_centered_on_the_frontier() -> None:
    assert prior_known(1_000, 1_000) == pytest.approx(0.5)
    assert prior_known(10, 1_000) > prior_known(100, 1_000) > 0.5 > prior_known(50_000, 1_000)
    assert prior_known(None, 1_000) == 0.5


@pytest.mark.parametrize(
    "overrides",
    [{"passive_confidence": 0}, {"reread_weight": 1.5}, {"weights": (1.0,)}, {"prior_slope": 0}],
)
def test_memory_policy_rejects_invalid_configuration(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MemoryPolicy(**overrides)  # type: ignore[arg-type]


# Independent vectors evaluated from the published FSRS-4.5 equations, not this module:
# https://github.com/open-spaced-repetition/awesome-fsrs/wiki/The-Algorithm#fsrs-45
# Full reviews only; Lang's partial reading evidence and frequency priors are adaptations.
def test_fsrs_45_reference_review_sequence() -> None:
    state = review(None, GOOD, T0)
    vectors = [
        (4, GOOD, 5.1618, 14.808100506496405),
        (14, GOOD, 5.1618, 38.71663295271267),
        (34, AGAIN, 6.901155, 4.655352765433932),
        (37, GOOD, 6.847234995, 10.573036466027194),
    ]
    for day, grade, difficulty, stability in vectors:
        state = review(state, grade, T0 + timedelta(days=day))
        assert state.difficulty == pytest.approx(difficulty, rel=1e-12)
        assert state.stability_days == pytest.approx(stability, rel=1e-12)
        assert state.last_review_at == T0 + timedelta(days=day)


@pytest.mark.parametrize("stability", [0.05, 1.0, 37.0, 3650.0])
@pytest.mark.parametrize("retention", [0.5, 0.8, 0.9, 0.99, 1.0])
def test_scheduled_interval_reaches_requested_recall(stability: float, retention: float) -> None:
    assert retrievability(stability, interval_days(stability, retention)) == pytest.approx(
        retention
    )


@pytest.mark.parametrize("retention", [0.0, -0.1, 1.01, float("nan"), float("inf")])
def test_invalid_retention_is_rejected(retention: float) -> None:
    with pytest.raises(ValueError, match="retention"):
        interval_days(1.0, retention)


@pytest.mark.parametrize("stability", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scheduling_stability_is_rejected(stability: float) -> None:
    with pytest.raises(ValueError, match="stability"):
        interval_days(stability)


@pytest.mark.parametrize(
    "name", ["horizon_days", "prior_slope", "minimum_stability_days", "maximum_stability_days"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_memory_policy_rejects_nonfinite_values(name: str, value: float) -> None:
    with pytest.raises(ValueError):
        MemoryPolicy(**{name: value})  # type: ignore[arg-type]


def test_memory_policy_rejects_nonfinite_weights() -> None:
    with pytest.raises(ValueError, match="finite weights"):
        MemoryPolicy(weights=(float("nan"), *FSRS_DEFAULT_WEIGHTS[1:]))
