from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.character_learning import CharacterLearningPolicy, rebuild_character_states
from server.learning import import_lesson, record_events
from server.models import CharacterState
from server.profile_state import update_profile


def _term(
    key: str,
    lemma: str,
    gloss: str,
    *,
    pos: str = "noun",
) -> dict[str, object]:
    return {
        "key": key,
        "lemma": lemma,
        "pos": pos,
        "gloss": gloss,
        "frequency_rank": 100,
    }


def _lesson(
    key: str,
    sentences: list[tuple[str, str, dict[str, object]]],
    *,
    learning_language: str = "zh-Hans",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "key": key,
        "title": f"Lesson {key}",
        "learning_language": learning_language,
        "translation_language": "en",
        "topic": "test",
        "level": "A1",
        "difficulty": 0.15,
        "blocks": [
            {
                "key": "body",
                "sentences": [
                    {
                        "key": sentence_key,
                        "runs": [
                            {"text": surface, "term": term},
                            {"text": "。"},
                        ],
                        "translation": f"Translation {index}",
                    }
                    for index, (sentence_key, surface, term) in enumerate(sentences, 1)
                ],
            }
        ],
        "target_term_keys": [],
        "metadata": {},
    }


def _states(db: Session) -> dict[str, CharacterState]:
    return {
        state.character: state
        for state in db.scalars(select(CharacterState).order_by(CharacterState.character)).all()
    }


def _snapshot(db: Session) -> list[tuple[object, ...]]:
    return [
        (
            state.learning_language,
            state.translation_language,
            state.character,
            state.alpha,
            state.beta,
            state.stability_days,
            state.qualified_exposures,
            state.inferred_failure_sessions,
            state.inferred_failure_mass,
            state.direct_successes,
            state.direct_failures,
            state.distinct_lessons,
            state.distinct_word_contexts,
            state.first_evidence_at,
            state.last_evidence_at,
            state.last_inferred_failure_at,
            state.next_due_at,
        )
        for state in db.scalars(
            select(CharacterState).order_by(
                CharacterState.learning_language,
                CharacterState.translation_language,
                CharacterState.character,
            )
        ).all()
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"new_context_success_bonus": -0.1},
        {"single_character_failure_mass": 0.6, "failure_cap_per_session": 0.5},
        {"multi_character_failure_mass": float("inf")},
        {"failure_cap_per_session": 1.5},
    ],
)
def test_character_policy_rejects_invalid_configuration(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        CharacterLearningPolicy(**overrides)


def test_passive_evidence_requires_qualified_unassisted_reading(
    db: Session,
    event_factory: Any,
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    lesson = import_lesson(
        db,
        _lesson(
            "qualified-reading",
            [
                ("revealed", "明", _term("zh:明:adjective", "明", "bright", pos="adjective")),
                ("translated", "月", _term("zh:月:noun", "月", "moon")),
                ("clean", "山", _term("zh:山:noun", "山", "mountain")),
            ],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="too-short",
                session_id="short",
                payload={"active_seconds": 29, "completion_ratio": 1},
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="reveal",
                session_id="qualified",
                payload={"term_key": "zh:明:adjective", "sentence_key": "revealed"},
                seconds=1,
            ),
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="sentence-help",
                session_id="qualified",
                payload={"scope": "sentence", "sentence_key": "translated"},
                seconds=2,
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="qualified",
                session_id="qualified",
                payload={"active_seconds": 30, "completion_ratio": 0.8},
                seconds=30,
            ),
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="full-help",
                session_id="full-help",
                payload={"scope": "lesson"},
                seconds=31,
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="full-help-completed",
                session_id="full-help",
                payload={"active_seconds": 60, "completion_ratio": 1},
                seconds=60,
            ),
        ],
    )

    rebuild_character_states(db)
    states = _states(db)

    assert states["明"].qualified_exposures == 0
    assert states["明"].inferred_failure_mass == pytest.approx(0.4)
    assert states["月"].qualified_exposures == 0
    assert states["月"].inferred_failure_mass == 0
    assert states["山"].qualified_exposures == 1
    # One clean reading in a new word context: passive confidence 0.45 x (1 + 0.25 bonus).
    assert states["山"].alpha == pytest.approx(2.5625)
    assert states["山"].distinct_lessons == 1
    assert states["山"].direct_successes == 0
    assert states["山"].direct_failures == 0


def test_reveals_are_deduplicated_weighted_and_capped_per_session(
    db: Session,
    event_factory: Any,
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    lesson = import_lesson(
        db,
        _lesson(
            "character-failures",
            [
                ("ming", "明", _term("zh:明:adjective", "明", "bright", pos="adjective")),
                ("moon", "明月", _term("zh:明月:noun", "明月", "bright moon")),
                ("fire-noun", "火", _term("zh:火:noun", "火", "fire")),
                ("fire-verb", "火", _term("zh:火:verb", "火", "to become angry", pos="verb")),
            ],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="ming-first",
                session_id="ming-session",
                payload={"term_key": "zh:明:adjective", "sentence_key": "ming"},
                seconds=1,
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="moon-second",
                session_id="moon-session",
                payload={"term_key": "zh:明月:noun", "sentence_key": "moon"},
                seconds=2,
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="fire-first",
                session_id="fire-session",
                payload={"term_key": "zh:火:noun", "sentence_key": "fire-noun"},
                seconds=3,
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="fire-duplicate",
                session_id="fire-session",
                payload={"term_key": "zh:火:noun", "sentence_key": "fire-noun"},
                seconds=4,
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="fire-other-word",
                session_id="fire-session",
                payload={"term_key": "zh:火:verb", "sentence_key": "fire-verb"},
                seconds=5,
            ),
        ],
    )

    rebuild_character_states(db)
    states = _states(db)

    assert states["明"].inferred_failure_mass > 0.5
    assert states["月"].inferred_failure_mass < 0.1
    assert states["明"].inferred_failure_mass + states["月"].inferred_failure_mass == pytest.approx(
        0.6
    )
    assert states["明"].inferred_failure_sessions == 2
    assert states["月"].inferred_failure_sessions == 1
    assert states["火"].inferred_failure_mass == pytest.approx(0.5)
    assert states["火"].inferred_failure_sessions == 1


def test_new_word_contexts_reinforce_more_than_repeated_contexts(
    db: Session,
    event_factory: Any,
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    bright = _term("zh:明:adjective", "明", "bright", pos="adjective")
    first = import_lesson(db, _lesson("context-first", [("first", "明", bright)]))
    repeated = import_lesson(
        db,
        _lesson(
            "context-repeated",
            [("repeated", "明", _term("zh:明:alias", "明", "bright", pos="adjective"))],
        ),
    )
    new_context = import_lesson(
        db,
        _lesson(
            "context-new",
            [("new", "明天", _term("zh:明天:noun", "明天", "tomorrow"))],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                first.id,
                "lesson.completed",
                event_id="first-completion",
                session_id="first",
                payload={"active_seconds": 30, "completion_ratio": 1},
                seconds=30,
            ),
            event_factory(
                repeated.id,
                "lesson.completed",
                event_id="repeated-completion",
                session_id="repeated",
                payload={"active_seconds": 30, "completion_ratio": 1},
                seconds=60,
            ),
            event_factory(
                new_context.id,
                "lesson.completed",
                event_id="new-completion",
                session_id="new",
                payload={"active_seconds": 30, "completion_ratio": 1},
                seconds=90,
            ),
        ],
    )

    rebuild_character_states(db)
    state = _states(db)["明"]

    assert state.qualified_exposures == 3
    assert state.alpha == pytest.approx(2 + 0.5625 + 0.45 + 0.5625)
    assert state.distinct_lessons == 3
    assert state.distinct_word_contexts == 2


def test_sixteen_massed_rereads_do_not_create_confident_character_recognition(
    db: Session,
    event_factory: Any,
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    lesson = import_lesson(
        db,
        _lesson(
            "common-character",
            [("demonstrative", "这", _term("zh:这:pronoun", "这", "this", pos="pronoun"))],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id=f"clean-{index}",
                session_id=f"session-{index}",
                payload={"active_seconds": 30, "completion_ratio": 1},
                seconds=30 + index,
            )
            for index in range(16)
        ],
    )

    state = _states(db)["这"]

    assert state.qualified_exposures == 1  # the other fifteen are rereads within seconds
    assert state.stability_days < 5
    assert state.memory_state is not None


def test_non_chinese_lessons_do_not_create_character_state(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    lesson = import_lesson(db, lesson_factory())
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="spanish-completed",
                payload={"active_seconds": 60, "completion_ratio": 1},
            )
        ],
    )

    assert rebuild_character_states(db) == []
    assert _states(db) == {}


def test_character_rebuild_is_deterministic_and_idempotent(
    db: Session,
    event_factory: Any,
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    lesson = import_lesson(
        db,
        _lesson(
            "deterministic",
            [
                ("river", "河", _term("zh:河:noun", "河", "river")),
                ("sea", "海", _term("zh:海:noun", "海", "sea")),
            ],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="river-reveal",
                payload={"term_key": "zh:河:noun", "sentence_key": "river"},
                seconds=1,
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="completed",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=45,
            ),
        ],
    )

    rebuild_character_states(db)
    first = _snapshot(db)
    rebuild_character_states(db)
    second = _snapshot(db)

    assert first == second
    assert all(value is None or isinstance(value, datetime) for row in second for value in row[-4:])
