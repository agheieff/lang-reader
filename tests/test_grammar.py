from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.agent_brief import build_agent_brief
from server.grammar import (
    GrammarLearningPolicy,
    get_grammar_state,
    grammar_marker_can_be_hidden,
    rebuild_grammar_states,
)
from server.learning import get_reader_state, import_lesson, record_events
from server.models import GrammarState, Interaction
from server.texts import get_text_detail


def _annotate(
    payload: dict[str, Any],
    sentence_index: int,
    construction_key: str,
    *,
    occurrence_key: str | None = None,
    note: str | None = None,
) -> str:
    sentence = payload["blocks"][0]["sentences"][sentence_index]
    occurrence_key = occurrence_key or f"{payload['key']}:grammar:{sentence_index}"
    sentence["grammar"] = [
        {
            "key": occurrence_key,
            "construction_key": construction_key,
            "run_start": 0,
            "run_end": len(sentence["runs"]),
            "note": note or f"Contextual note for {construction_key}.",
        }
    ]
    return occurrence_key


def _states(db: Session) -> dict[str, GrammarState]:
    return {state.construction_key: state for state in db.scalars(select(GrammarState)).all()}


def _snapshot(db: Session) -> list[tuple[Any, ...]]:
    return [
        (
            state.learning_language,
            state.translation_language,
            state.construction_key,
            state.alpha,
            state.beta,
            state.stability_days,
            state.qualified_exposures,
            state.explicit_help_failures,
            state.inferred_difficulty_signals,
            state.distinct_lessons,
            state.first_seen_at,
            state.last_seen_at,
            state.last_helped_at,
            state.next_due_at,
        )
        for state in db.scalars(
            select(GrammarState).order_by(
                GrammarState.learning_language,
                GrammarState.translation_language,
                GrammarState.construction_key,
            )
        ).all()
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"clean_success_weight": -0.1},
        {"new_lesson_success_bonus": float("inf")},
        {"explicit_help_weight": -0.1},
        {"inferred_difficulty_weight": -0.1},
        {"max_inferred_vocabulary_reveals": -1},
        {"max_inferred_vocabulary_reveals": 1.5},
        {"explicit_help_due_days": -0.1},
        {"inferred_difficulty_due_days": -0.1},
        {"stability_gain_days_per_success_mass": -0.1},
    ],
)
def test_grammar_policy_rejects_invalid_configuration(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        GrammarLearningPolicy(**overrides)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, True),
        ({"qualified_exposures": 3}, False),
        ({"exposed_lesson_count": 2}, False),
        ({"mastery": 0.799}, False),
        ({"stability_days": 6.999}, False),
        ({"qualified_exposures": 6, "counted_help_sessions": 2}, True),
        ({"counted_help_sessions": 2}, False),
        ({"inferred_difficulty_signals": 2}, False),
    ],
)
def test_grammar_marker_comfort_policy_is_conservative_and_inclusive(
    override: dict[str, int | float], expected: bool
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evidence: dict[str, Any] = {
        "qualified_exposures": 4,
        "exposed_lesson_count": 3,
        "mastery": 0.8,
        "stability_days": 7.0,
        "counted_help_sessions": 0,
        "inferred_difficulty_signals": 0,
        "next_due_at": now + timedelta(days=1),
    }
    evidence.update(override)

    assert grammar_marker_can_be_hidden(**evidence, now=now) is expected


def test_grammar_marker_returns_when_due_and_stays_visible_without_a_due_date() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def can_hide(next_due_at: datetime | None) -> bool:
        return grammar_marker_can_be_hidden(
            qualified_exposures=6,
            exposed_lesson_count=3,
            mastery=0.82,
            stability_days=8.0,
            counted_help_sessions=1,
            inferred_difficulty_signals=0,
            next_due_at=next_due_at,
            now=now,
        )

    assert can_hide(now + timedelta(seconds=1))
    assert not can_hide(now)
    assert not can_hide(now - timedelta(seconds=1))
    assert not can_hide(None)


def test_reader_and_preview_expose_comfortable_grammar_without_changing_annotations(
    db: Session, lesson_factory: Any, event_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "server.grammar.utc_now",
        lambda: datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    construction_key = "es:present-indicative"
    overdue_key = "es:ser-estar"

    def import_evidence_lessons(key: str, construction: str) -> list[Any]:
        lessons = []
        for index in range(3):
            payload = lesson_factory(key=f"{key}-{index}", targets=[])
            _annotate(payload, 0, construction)
            lessons.append(import_lesson(db, payload))
        return lessons

    evidence_lessons = import_evidence_lessons("grammar-comfort-evidence", construction_key)
    overdue_lessons = import_evidence_lessons("grammar-overdue-evidence", overdue_key)

    events = []
    for prefix, lessons, base_seconds in (
        ("future", evidence_lessons, 0),
        ("overdue", overdue_lessons, -8 * 365 * 86_400),
    ):
        for index in range(7):
            lesson = lessons[index % len(lessons)]
            events.append(
                event_factory(
                    lesson.id,
                    "lesson.completed",
                    event_id=f"grammar-{prefix}-completion-{index}",
                    session_id=f"grammar-{prefix}-session-{index}",
                    payload={"active_seconds": 45, "completion_ratio": 1},
                    seconds=base_seconds + index * 30 * 86_400,
                )
            )
    record_events(db, events)

    payload = lesson_factory(key="grammar-comfort-reader", targets=[])
    occurrence_key = _annotate(payload, 0, construction_key)
    overdue_occurrence_key = _annotate(payload, 1, overdue_key)
    lesson = import_lesson(db, payload)
    interaction_count = db.scalar(select(func.count()).select_from(Interaction))

    reader = get_reader_state(db, lesson_id=lesson.id)
    preview = get_text_detail(db, lesson.id)

    assert reader.comfortable_grammar_construction_keys == [construction_key]
    assert preview.comfortable_grammar_construction_keys == [construction_key]
    assert reader.lesson is not None
    assert occurrence_key in reader.lesson.grammar_occurrences()
    assert overdue_occurrence_key in reader.lesson.grammar_occurrences()
    assert occurrence_key in preview.lesson.grammar_occurrences()
    assert overdue_occurrence_key in preview.lesson.grammar_occurrences()
    assert db.scalar(select(func.count()).select_from(Interaction)) == interaction_count


def test_explicit_help_is_stronger_than_inference_and_repeats_are_capped(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    payload = lesson_factory(key="grammar-signals", targets=[])
    explicit_occurrence = _annotate(payload, 0, "es:present-indicative")
    _annotate(payload, 1, "es:ser-estar")
    _annotate(payload, 2, "es:direct-object-clitic")
    lesson = import_lesson(db, payload)

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="grammar-help-one",
                payload={
                    "scope": "grammar",
                    "sentence_key": "sentence-1",
                    "occurrence_key": explicit_occurrence,
                    "construction_key": "es:present-indicative",
                },
                seconds=1,
            ),
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="grammar-help-repeat",
                payload={
                    "scope": "grammar",
                    "sentence_key": "sentence-1",
                    "occurrence_key": explicit_occurrence,
                    "construction_key": "es:present-indicative",
                },
                seconds=2,
            ),
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="sentence-help",
                payload={"scope": "sentence", "sentence_key": "sentence-2"},
                seconds=3,
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="qualified-completion",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=45,
            ),
        ],
    )
    rebuild_grammar_states(db)
    states = _states(db)

    explicit = states["es:present-indicative"]
    inferred = states["es:ser-estar"]
    successful = states["es:direct-object-clitic"]
    assert explicit.explicit_help_failures == 1
    assert explicit.inferred_difficulty_signals == 0
    assert explicit.beta == pytest.approx(3.0)
    assert inferred.explicit_help_failures == 0
    assert inferred.inferred_difficulty_signals == 1
    assert inferred.beta == pytest.approx(2.35)
    assert successful.qualified_exposures == 1
    assert successful.alpha == pytest.approx(3.25)
    assert explicit.mastery < inferred.mastery < 0.5 < successful.mastery
    assert db.scalar(select(func.count()).select_from(Interaction)) == 4

    view = get_grammar_state(db)
    by_key = {construction.key: construction for construction in view.constructions}
    assert by_key["es:present-indicative"].raw_help_count == 2
    assert by_key["es:present-indicative"].counted_help_sessions == 1


def test_sentence_translation_with_vocabulary_reveals_is_not_grammar_failure(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    payload = lesson_factory(key="vocabulary-explains-translation", targets=[])
    first_sentence = payload["blocks"][0]["sentences"][0]
    other_sentences = payload["blocks"][0]["sentences"][1:]
    for sentence in other_sentences:
        first_sentence["runs"].extend([{"text": " "}, deepcopy(sentence["runs"][0])])
    _annotate(payload, 0, "es:present-indicative")
    lesson = import_lesson(db, payload)

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="first-vocabulary-help",
                payload={"term_key": "es:manana:NOUN"},
                seconds=1,
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="second-vocabulary-help",
                payload={"term_key": "es:corazon:NOUN"},
                seconds=2,
            ),
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="translated-after-word-help",
                payload={"scope": "sentence", "sentence_key": "sentence-1"},
                seconds=3,
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="completion-after-translation",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=45,
            ),
        ],
    )
    rebuild_grammar_states(db)
    state = _states(db)["es:present-indicative"]

    assert state.inferred_difficulty_signals == 0
    assert state.qualified_exposures == 0
    assert state.mastery == pytest.approx(0.5)
    assert state.first_seen_at is None


def test_reader_restores_grammar_help_without_marking_sentence_translation(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    payload = lesson_factory(key="reader-grammar", targets=[])
    occurrence_key = _annotate(payload, 0, "es:present-indicative", note=None)
    lesson = import_lesson(db, payload)
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="reader-grammar-help",
                payload={
                    "scope": "grammar",
                    "sentence_key": "sentence-1",
                    "occurrence_key": occurrence_key,
                    "construction_key": "es:present-indicative",
                },
            )
        ],
    )

    reader = get_reader_state(db, lesson_id=lesson.id)

    assert reader.progress.revealed_grammar_occurrence_keys == [occurrence_key]
    assert reader.progress.revealed_sentence_keys == []
    assert [entry.key for entry in reader.grammar_catalog] == ["es:present-indicative"]


def test_agent_brief_offers_catalog_choices_and_cools_unread_annotations(
    db: Session, lesson_factory: Any
) -> None:
    initial = build_agent_brief(db)
    catalog_keys = {entry.key for entry in initial.grammar_catalog}
    priority_keys = {entry.key for entry in initial.priority_grammar}
    assert "es:present-indicative" in catalog_keys
    assert priority_keys
    assert priority_keys <= catalog_keys

    payload = lesson_factory(key="unread-grammar", targets=[])
    _annotate(payload, 0, "es:present-indicative")
    import_lesson(db, payload)
    brief = build_agent_brief(db)

    assert brief.recent_lessons[0].grammar_keys == ["es:present-indicative"]
    assert "es:present-indicative" not in {entry.key for entry in brief.priority_grammar}


def test_event_validation_requires_the_exact_grammar_occurrence_reference(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    payload = lesson_factory(key="grammar-reference", targets=[])
    occurrence_key = _annotate(payload, 0, "es:present-indicative")
    lesson = import_lesson(db, payload)

    with pytest.raises(ValueError, match="does not belong to sentence"):
        record_events(
            db,
            [
                event_factory(
                    lesson.id,
                    "translation.revealed",
                    event_id="wrong-grammar-sentence",
                    payload={
                        "scope": "grammar",
                        "sentence_key": "sentence-2",
                        "occurrence_key": occurrence_key,
                        "construction_key": "es:present-indicative",
                    },
                )
            ],
        )


def test_full_translation_suppresses_passive_grammar_success(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    payload = lesson_factory(key="full-translation", targets=[])
    _annotate(payload, 0, "es:present-indicative")
    lesson = import_lesson(db, payload)
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="full-translation",
                payload={"scope": "lesson"},
                seconds=1,
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="translated-completion",
                payload={"active_seconds": 60, "completion_ratio": 1},
                seconds=60,
            ),
        ],
    )
    rebuild_grammar_states(db)
    state = _states(db)["es:present-indicative"]

    assert state.alpha == pytest.approx(2.0)
    assert state.beta == pytest.approx(2.0)
    assert state.qualified_exposures == 0
    assert state.first_seen_at is None


def test_delayed_grammar_success_builds_more_stability_and_replays_exactly(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lessons = []
    specs = (
        ("fast-first", "es:present-indicative"),
        ("fast-second", "es:present-indicative"),
        ("delayed-first", "es:ser-estar"),
        ("delayed-second", "es:ser-estar"),
    )
    for key, construction_key in specs:
        payload = lesson_factory(key=key, targets=[])
        _annotate(payload, 0, construction_key)
        lessons.append(import_lesson(db, payload))

    record_events(
        db,
        [
            event_factory(
                lessons[3].id,
                "lesson.completed",
                event_id="delayed-second-success",
                session_id="delayed-second",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=30 * 86_400,
            ),
            event_factory(
                lessons[0].id,
                "lesson.completed",
                event_id="fast-first-success",
                session_id="fast-first",
                payload={"active_seconds": 45, "completion_ratio": 1},
            ),
            event_factory(
                lessons[2].id,
                "lesson.completed",
                event_id="delayed-first-success",
                session_id="delayed-first",
                payload={"active_seconds": 45, "completion_ratio": 1},
            ),
            event_factory(
                lessons[1].id,
                "lesson.completed",
                event_id="fast-second-success",
                session_id="fast-second",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=3_600,
            ),
        ],
    )
    rebuild_grammar_states(db)
    states = _states(db)
    fast = states["es:present-indicative"]
    delayed = states["es:ser-estar"]

    assert fast.alpha == pytest.approx(delayed.alpha)
    assert fast.qualified_exposures == delayed.qualified_exposures == 2
    assert delayed.stability_days > fast.stability_days
    before = _snapshot(db)
    rebuild_grammar_states(db)
    assert _snapshot(db) == before


def test_varied_clean_grammar_sessions_build_confidence_and_stability(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    construction_key = "es:present-indicative"
    lessons = []
    for index in range(8):
        payload = lesson_factory(key=f"varied-grammar-{index}", targets=[])
        _annotate(payload, 0, construction_key)
        lessons.append(import_lesson(db, payload))

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id=f"varied-grammar-completion-{index}",
                session_id=f"varied-grammar-session-{index}",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=index * 86_400,
            )
            for index, lesson in enumerate(lessons)
        ],
    )
    rebuild_grammar_states(db)
    state = _states(db)[construction_key]

    assert state.qualified_exposures == 8
    assert state.alpha == pytest.approx(12.0)
    assert state.mastery == pytest.approx(6 / 7)
    assert state.stability_days > 8


def test_rereading_one_grammar_context_gets_only_one_variety_bonus(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    construction_key = "es:present-indicative"
    payload = lesson_factory(key="same-grammar-context", targets=[])
    _annotate(payload, 0, construction_key)
    lesson = import_lesson(db, payload)
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id=f"same-context-{index}",
                session_id=f"same-context-session-{index}",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=index * 86_400,
            )
            for index in range(2)
        ],
    )
    rebuild_grammar_states(db)
    state = _states(db)[construction_key]

    assert state.qualified_exposures == 2
    assert state.alpha == pytest.approx(4.25)


def test_delayed_explicit_help_is_less_surprising_than_immediate_help(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lessons = []
    occurrences: list[str] = []
    specs = (
        ("fast-known", "es:present-indicative"),
        ("fast-help", "es:present-indicative"),
        ("delayed-known", "es:ser-estar"),
        ("delayed-help", "es:ser-estar"),
    )
    for key, construction_key in specs:
        payload = lesson_factory(key=key, targets=[])
        occurrences.append(_annotate(payload, 0, construction_key))
        lessons.append(import_lesson(db, payload))

    record_events(
        db,
        [
            event_factory(
                lessons[0].id,
                "lesson.completed",
                event_id="fast-initial-success",
                session_id="fast-initial",
                payload={"active_seconds": 45, "completion_ratio": 1},
            ),
            event_factory(
                lessons[2].id,
                "lesson.completed",
                event_id="delayed-initial-success",
                session_id="delayed-initial",
                payload={"active_seconds": 45, "completion_ratio": 1},
            ),
            event_factory(
                lessons[1].id,
                "translation.revealed",
                event_id="fast-explicit-help",
                session_id="fast-help",
                payload={
                    "scope": "grammar",
                    "sentence_key": "sentence-1",
                    "occurrence_key": occurrences[1],
                    "construction_key": "es:present-indicative",
                },
                seconds=3_600,
            ),
            event_factory(
                lessons[3].id,
                "translation.revealed",
                event_id="delayed-explicit-help",
                session_id="delayed-help",
                payload={
                    "scope": "grammar",
                    "sentence_key": "sentence-1",
                    "occurrence_key": occurrences[3],
                    "construction_key": "es:ser-estar",
                },
                seconds=30 * 86_400,
            ),
        ],
    )
    rebuild_grammar_states(db)
    states = _states(db)
    fast = states["es:present-indicative"]
    delayed = states["es:ser-estar"]

    assert fast.alpha == pytest.approx(delayed.alpha)
    assert fast.beta == pytest.approx(delayed.beta)
    assert fast.explicit_help_failures == delayed.explicit_help_failures == 1
    assert delayed.stability_days > fast.stability_days


def test_grammar_list_only_uses_opened_lessons_and_includes_examples(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    unopened_payload = lesson_factory(key="prepared-only", targets=[])
    _annotate(unopened_payload, 0, "es:present-indicative")
    import_lesson(db, unopened_payload)

    opened_payload = lesson_factory(key="opened-grammar", targets=[])
    _annotate(
        opened_payload,
        0,
        "es:ser-estar",
        note="This predicate describes a temporary state.",
    )
    opened = import_lesson(db, opened_payload)
    record_events(
        db,
        [
            event_factory(
                opened.id,
                "lesson.started",
                event_id="opened-lesson",
                session_id="reading-session",
            )
        ],
    )

    state = get_grammar_state(db)
    assert [construction.key for construction in state.constructions] == ["es:ser-estar"]
    construction = state.constructions[0]
    assert construction.occurrence_count == 1
    assert construction.exposed_lesson_count == 1
    assert construction.mastery == pytest.approx(0.5)
    assert len(construction.examples) == 1
    assert construction.examples[0].title == "Lesson opened-grammar"
    assert construction.examples[0].note == "This predicate describes a temporary state."
