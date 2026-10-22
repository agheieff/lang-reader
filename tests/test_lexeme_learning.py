from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.clock import as_utc
from server.learning import import_lesson, record_events
from server.learning_policy import TERM_BAND_POLICY
from server.lexeme_learning import DEFAULT_LEXEME_LEARNING_POLICY, LexemeLearningPolicy
from server.memory_model import knowledge
from server.models import LexemeState

BASE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
DAY = 86_400


@pytest.mark.parametrize("policy", [{"frontier_scale": 0}, {"frontier_scale": float("nan")}])
def test_lexeme_policy_rejects_invalid_configuration(policy: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LexemeLearningPolicy(**policy)


def _knowledge_at(state: LexemeState, at: datetime) -> float:
    return knowledge(
        state.memory_state,
        prior=state.prior_known,
        at=at,
        policy=DEFAULT_LEXEME_LEARNING_POLICY.memory,
    )


def _completion(event_factory: Any, lesson_id: int, index: int, seconds: int) -> dict[str, Any]:
    return event_factory(
        lesson_id,
        "lesson.completed",
        event_id=f"clean-{lesson_id}-{index}",
        session_id=f"session-{lesson_id}-{index}",
        payload={"active_seconds": 30, "completion_ratio": 1},
        seconds=seconds,
    )


def test_massed_rereads_do_not_create_confident_recognition(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    # A mid-frequency word, so the frequency prior does not already imply durable knowledge.
    term = ("es:rincón:NOUN", "rincón", "rincón", "NOUN", "corner", 8_000)
    lesson = import_lesson(db, lesson_factory(key="mid-word", terms=[term], targets=[]))
    record_events(
        db, [_completion(event_factory, lesson.id, index, 30 + index) for index in range(10)]
    )

    state = db.scalar(select(LexemeState).where(LexemeState.term_key == term[0]))

    assert state is not None
    assert state.qualified_exposures == 1  # rereads are not new exposures
    assert state.stability_days < 5
    assert _knowledge_at(state, BASE + timedelta(minutes=1)) < TERM_BAND_POLICY.familiar_min_mastery


def test_spaced_clean_readings_grow_durable_recognition(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    term = ("es:comprobar:VERB", "comprobar", "comprobar", "VERB", "check", 3_000)
    days = (0, 2, 6, 14, 30, 60)
    stabilities: list[float] = []
    for index, day in enumerate(days):
        lesson = import_lesson(db, lesson_factory(key=f"spaced-{index}", terms=[term], targets=[]))
        record_events(db, [_completion(event_factory, lesson.id, index, day * DAY + 60)])
        state = db.scalar(select(LexemeState).where(LexemeState.term_key == term[0]))
        assert state is not None
        stabilities.append(state.stability_days)

    assert stabilities == sorted(stabilities)
    assert stabilities[-1] > 30
    assert state.qualified_exposures == len(days)
    assert _knowledge_at(state, BASE + timedelta(days=60)) >= TERM_BAND_POLICY.mastered_min_mastery


def test_a_reveal_is_a_lapse_that_resets_durable_recognition(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    term = ("es:olvidar:VERB", "olvidar", "olvidar", "VERB", "forget", 2_000)
    lessons = [
        import_lesson(db, lesson_factory(key=f"lapse-{index}", terms=[term], targets=[]))
        for index in range(4)
    ]
    record_events(
        db,
        [
            _completion(event_factory, lesson.id, index, day * DAY)
            for index, (lesson, day) in enumerate(zip(lessons[:3], (0, 3, 10), strict=True))
        ],
    )
    before = db.scalar(select(LexemeState).where(LexemeState.term_key == term[0]))
    assert before is not None
    stability_before = before.stability_days

    record_events(
        db,
        [
            event_factory(
                lessons[3].id,
                "term.revealed",
                event_id="lapse",
                session_id="lapse-session",
                payload={"term_key": term[0]},
                seconds=40 * DAY,
            )
        ],
    )
    after = db.scalar(select(LexemeState).where(LexemeState.term_key == term[0]))

    assert after is not None
    assert after.stability_days < stability_before
    assert after.reveal_failures == 1
    assert after.next_due_at is not None
    due = as_utc(after.next_due_at)
    assert due - (BASE + timedelta(days=40)) < timedelta(days=stability_before)


def test_unseen_words_use_a_frequency_prior(db: Session, lesson_factory: Any) -> None:
    common = ("es:el:DET", "el", "el", "DET", "the", 1)
    rare = ("es:ornitorrinco:NOUN", "ornitorrinco", "ornitorrinco", "NOUN", "platypus", 40_000)
    import_lesson(db, lesson_factory(key="prior", terms=[common, rare], targets=[]))
    states = {state.term_key: state for state in db.scalars(select(LexemeState))}

    assert states[common[0]].memory_state is None
    assert states[common[0]].mastery > 0.9
    assert states[rare[0]].mastery < 0.2
