from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.agent_brief import build_agent_brief, build_generation_target_policy
from server.learning import (
    EventConflictError,
    LessonConflictError,
    get_reader_state,
    import_lesson,
    record_events,
    validate_lesson,
)
from server.lexeme_learning import rebuild_lexeme_states
from server.memory_model import FSRS_DEFAULT_WEIGHTS
from server.models import Interaction, Lesson, LexemeState, ProficiencyState
from server.profile_state import update_profile
from server.schemas import ReaderState


def _states(db: Session) -> dict[str, LexemeState]:
    return {state.term_key: state for state in db.scalars(select(LexemeState)).all()}


def _state_snapshot(db: Session) -> list[tuple[Any, ...]]:
    states = db.scalars(
        select(LexemeState).order_by(
            LexemeState.learning_language,
            LexemeState.translation_language,
            LexemeState.term_key,
        )
    ).all()
    return [
        (
            state.learning_language,
            state.translation_language,
            state.term_key,
            state.alpha,
            state.beta,
            state.stability_days,
            state.qualified_exposures,
            state.reveal_failures,
            state.distinct_lessons,
            state.first_seen_at,
            state.last_seen_at,
            state.last_revealed_at,
            state.next_due_at,
        )
        for state in states
    ]


def test_schema_preserves_unicode_and_xss_text(lesson_factory: Any) -> None:
    payload = lesson_factory()
    document = validate_lesson(payload)

    first = document.blocks[0].sentences[0]
    assert first.runs[0].text == "<script>alert('x')</script> 🌍 "
    assert first.runs[1].text == "mañana🧠"
    assert first.text.startswith("<script>")
    assert document.learning_language == "es-ES"
    assert document.level == "A1"
    assert document.difficulty == 0.15

    unknown_target = deepcopy(payload)
    unknown_target["target_term_keys"] = ["missing"]
    with pytest.raises(ValidationError, match="unknown terms"):
        validate_lesson(unknown_target)

    blank_term_run = deepcopy(payload)
    blank_term_run["blocks"][0]["sentences"][0]["runs"][1]["text"] = "   "
    with pytest.raises(ValidationError, match="visible text"):
        validate_lesson(blank_term_run)

    invalid_difficulty = deepcopy(payload)
    invalid_difficulty["difficulty"] = 1.01
    with pytest.raises(ValidationError):
        validate_lesson(invalid_difficulty)


def test_import_is_idempotent_and_rejects_conflicts(db: Session, lesson_factory: Any) -> None:
    payload = lesson_factory()
    lesson = import_lesson(db, payload)
    repeated = import_lesson(db, deepcopy(payload))

    assert repeated.id == lesson.id
    assert db.scalar(select(func.count()).select_from(Lesson)) == 1

    changed = deepcopy(payload)
    changed["title"] = "Different content"
    with pytest.raises(LessonConflictError, match="already exists"):
        import_lesson(db, changed)

    wrong_language = lesson_factory(key="wrong-language", learning_language="fr")
    with pytest.raises(ValueError, match="languages must match"):
        import_lesson(db, wrong_language)

    conflicting_term = lesson_factory(key="conflicting-term")
    conflicting_term["blocks"][0]["sentences"][0]["runs"][1]["term"]["gloss"] = "later"
    with pytest.raises(ValueError, match="conflicting definitions"):
        import_lesson(db, conflicting_term)


def test_event_idempotency_and_conflicting_reuse(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    event = event_factory(
        lesson.id,
        "term.revealed",
        event_id="reveal-1",
        payload={"term_key": "es:manana:NOUN"},
    )

    first = record_events(db, [event, deepcopy(event)])
    replay = record_events(db, [deepcopy(event)])

    assert first.accepted == 1
    assert first.duplicates == 1
    assert replay.accepted == 0
    assert replay.duplicates == 1
    assert db.scalar(select(func.count()).select_from(Interaction)) == 1

    conflict = deepcopy(event)
    conflict["payload"] = {"term_key": "es:corazon:NOUN"}
    with pytest.raises(EventConflictError, match="different content"):
        record_events(db, [conflict])


def test_passive_evidence_requires_a_qualified_completion(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="short-completion",
                session_id="short",
                payload={"active_seconds": 29, "completion_ratio": 1},
            )
        ],
    )
    assert all(state.qualified_exposures == 0 for state in _states(db).values())

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="qualified-completion",
                session_id="qualified",
                payload={"active_seconds": 30, "completion_ratio": 0.8},
            )
        ],
    )
    assert all(state.qualified_exposures == 1 for state in _states(db).values())


def test_reveals_and_sentence_translation_exclude_passive_evidence(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    events = [
        event_factory(
            lesson.id,
            "term.revealed",
            event_id="reveal-1",
            payload={"term_key": "es:manana:NOUN"},
            seconds=1,
        ),
        event_factory(
            lesson.id,
            "term.revealed",
            event_id="reveal-2",
            payload={"term_key": "es:manana:NOUN"},
            seconds=2,
        ),
        event_factory(
            lesson.id,
            "translation.revealed",
            event_id="translate-sentence",
            payload={"scope": "sentence", "sentence_key": "sentence-2"},
            seconds=3,
        ),
        event_factory(
            lesson.id,
            "lesson.completed",
            event_id="complete",
            payload={"active_seconds": 45, "completion_ratio": 1},
            seconds=45,
        ),
    ]
    record_events(db, events)
    states = _states(db)

    assert states["es:manana:NOUN"].reveal_failures == 1
    assert states["es:manana:NOUN"].qualified_exposures == 0
    assert states["es:corazon:NOUN"].qualified_exposures == 0
    assert states["es:mundo:NOUN"].qualified_exposures == 1


def test_annotated_title_terms_participate_in_event_validation_and_evidence(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    payload = lesson_factory(targets=[])
    payload["title"] = "Mi día"
    payload["title_sentence"] = {
        "key": "lesson-title",
        "runs": [
            {
                "text": "Mi",
                "term": {
                    "key": "es:mi:DET",
                    "lemma": "mi",
                    "pos": "DET",
                    "gloss": "my",
                    "frequency_rank": 30,
                },
            },
            {"text": " "},
            {
                "text": "día",
                "term": {
                    "key": "es:dia:NOUN",
                    "lemma": "día",
                    "pos": "NOUN",
                    "gloss": "day",
                    "frequency_rank": 100,
                },
            },
        ],
        "translation": "My day",
    }
    lesson = import_lesson(db, payload)

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="title-term-reveal",
                payload={"term_key": "es:mi:DET"},
            ),
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="title-translation",
                payload={"scope": "sentence", "sentence_key": "lesson-title"},
                seconds=1,
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="annotated-title-completion",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=45,
            ),
        ],
    )
    states = _states(db)

    assert states["es:mi:DET"].reveal_failures == 1
    assert states["es:mi:DET"].qualified_exposures == 0
    assert states["es:dia:NOUN"].qualified_exposures == 0
    assert states["es:mundo:NOUN"].qualified_exposures == 1


def test_reveals_are_lapses_across_sessions_and_capped_within_a_session(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    # A rare word: its frequency prior is negligible, so the first lapse starts from FSRS values.
    term = ("es:comprobar:VERB", "comprobar", "comprobar", "VERB", "check", 40_000)
    first_lesson = import_lesson(
        db,
        lesson_factory(key="graded-reveal-one", terms=[term], targets=[]),
    )
    second_lesson = import_lesson(
        db,
        lesson_factory(key="graded-reveal-two", terms=[term], targets=[]),
    )

    record_events(
        db,
        [
            event_factory(
                first_lesson.id,
                "term.revealed",
                event_id="first-check",
                session_id="first-session",
                payload={"term_key": term[0]},
            ),
            event_factory(
                first_lesson.id,
                "term.revealed",
                event_id="repeat-check",
                session_id="first-session",
                payload={"term_key": term[0]},
                seconds=1,
            ),
        ],
    )
    first = _states(db)[term[0]]
    assert first.reveal_failures == 1
    assert first.beta == pytest.approx(3.0)
    assert first.stability_days == pytest.approx(FSRS_DEFAULT_WEIGHTS[0])

    record_events(
        db,
        [
            event_factory(
                first_lesson.id,
                "term.revealed",
                event_id="second-session-check",
                session_id="second-session",
                payload={"term_key": term[0]},
                seconds=2,
            ),
            event_factory(
                second_lesson.id,
                "term.revealed",
                event_id="second-lesson-check",
                session_id="third-session",
                payload={"term_key": term[0]},
                seconds=3,
            ),
        ],
    )
    repeated = _states(db)[term[0]]
    assert repeated.reveal_failures == 3
    assert repeated.beta == pytest.approx(5.0)
    assert repeated.stability_days <= first.stability_days
    assert (
        db.scalar(
            select(func.count())
            .select_from(Interaction)
            .where(Interaction.event_type == "term.revealed")
        )
        == 4
    )


def test_delayed_passive_success_increases_stability_more_and_replays_exactly(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    fast = ("es:fast:NOUN", "rápido", "rápido", "NOUN", "fast", 500)
    delayed = ("es:delayed:NOUN", "tardío", "tardío", "NOUN", "delayed", 500)
    fast_lessons = [
        import_lesson(
            db,
            lesson_factory(key=f"fast-success-{index}", terms=[fast], targets=[]),
        )
        for index in range(2)
    ]
    delayed_lessons = [
        import_lesson(
            db,
            lesson_factory(key=f"delayed-success-{index}", terms=[delayed], targets=[]),
        )
        for index in range(2)
    ]
    completion = {"active_seconds": 45, "completion_ratio": 1}

    # Deliberately submit out of timestamp order; rebuild order is event-time deterministic.
    record_events(
        db,
        [
            event_factory(
                delayed_lessons[1].id,
                "lesson.completed",
                event_id="delayed-success-later",
                session_id="delayed-later",
                payload=completion,
                seconds=30 * 86_400,
            ),
            event_factory(
                fast_lessons[0].id,
                "lesson.completed",
                event_id="fast-success-first",
                session_id="fast-first",
                payload=completion,
            ),
            event_factory(
                delayed_lessons[0].id,
                "lesson.completed",
                event_id="delayed-success-first",
                session_id="delayed-first",
                payload=completion,
            ),
            event_factory(
                fast_lessons[1].id,
                "lesson.completed",
                event_id="fast-success-later",
                session_id="fast-later",
                payload=completion,
                seconds=3_600,
            ),
        ],
    )
    states = _states(db)

    assert states[fast[0]].qualified_exposures == states[delayed[0]].qualified_exposures == 2
    assert states[fast[0]].alpha == pytest.approx(states[delayed[0]].alpha)
    assert states[delayed[0]].stability_days > states[fast[0]].stability_days
    snapshot = _state_snapshot(db)
    rebuild_lexeme_states(db)
    assert _state_snapshot(db) == snapshot


def test_delayed_reveal_is_less_surprising_than_a_near_immediate_reveal(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    fast = ("es:fast-failure:NOUN", "pronto", "pronto", "NOUN", "soon", 500)
    delayed = ("es:delayed-failure:NOUN", "después", "después", "NOUN", "later", 500)
    fast_success = import_lesson(
        db,
        lesson_factory(key="fast-failure-prior", terms=[fast], targets=[]),
    )
    fast_failure = import_lesson(
        db,
        lesson_factory(key="fast-failure-result", terms=[fast], targets=[]),
    )
    delayed_success = import_lesson(
        db,
        lesson_factory(key="delayed-failure-prior", terms=[delayed], targets=[]),
    )
    delayed_failure = import_lesson(
        db,
        lesson_factory(key="delayed-failure-result", terms=[delayed], targets=[]),
    )
    completion = {"active_seconds": 45, "completion_ratio": 1}

    record_events(
        db,
        [
            event_factory(
                fast_success.id,
                "lesson.completed",
                event_id="fast-failure-success",
                session_id="fast-success",
                payload=completion,
            ),
            event_factory(
                delayed_success.id,
                "lesson.completed",
                event_id="delayed-failure-success",
                session_id="delayed-success",
                payload=completion,
            ),
            event_factory(
                fast_failure.id,
                "term.revealed",
                event_id="fast-failure-reveal",
                session_id="fast-reveal",
                payload={"term_key": fast[0]},
                seconds=3_600,
            ),
            event_factory(
                delayed_failure.id,
                "term.revealed",
                event_id="delayed-failure-reveal",
                session_id="delayed-reveal",
                payload={"term_key": delayed[0]},
                seconds=30 * 86_400,
            ),
        ],
    )
    states = _states(db)

    assert states[fast[0]].reveal_failures == states[delayed[0]].reveal_failures == 1
    assert states[fast[0]].beta == pytest.approx(states[delayed[0]].beta)
    assert states[delayed[0]].stability_days > states[fast[0]].stability_days


def test_language_units_filter_productive_spans_and_resolve_exact_aliases(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    update_profile(
        db,
        {
            "learning_language": "zh-Hans",
            "translation_language": "en",
        },
    )
    payload = lesson_factory(
        key="zh-learning-units",
        learning_language="zh-Hans",
        terms=[
            ("zh:yi-ge:phrase", "一个", "一个", "quantity_phrase", "one item", 10),
            ("zh:yi:num", "一", "一", "numeral", "one", 2),
            ("zh:ge:clf", "个", "个", "classifier", "general classifier", 20),
            ("zh:yiqi:adv", "一起", "一起", "adverb", "together", 500),
        ],
        targets=["zh:yi-ge:phrase"],
    )
    lesson = import_lesson(db, payload)

    states = _states(db)
    assert "zh:yi-ge:phrase" not in states
    assert set(states) == {"zh:yi:num", "zh:ge:clf", "zh:yiqi:adv"}
    reader = get_reader_state(db, lesson.id)
    assert reader.term_bands["zh:yi-ge:phrase"] == "incidental"

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="reveal-productive-span",
                payload={"term_key": "zh:yi-ge:phrase"},
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="complete-productivity-test",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=45,
            ),
        ],
    )

    states = _states(db)
    assert "zh:yi-ge:phrase" not in states
    assert states["zh:yi:num"].qualified_exposures == 0
    assert states["zh:ge:clf"].qualified_exposures == 0
    assert states["zh:yiqi:adv"].qualified_exposures == 1
    assert states["zh:yi:num"].reveal_failures == pytest.approx(0.5)
    assert states["zh:ge:clf"].reveal_failures == pytest.approx(0.5)
    assert states["zh:yiqi:adv"].reveal_failures == 0
    stored = db.scalar(select(Interaction).where(Interaction.event_id == "reveal-productive-span"))
    assert stored is not None
    assert stored.payload == {"term_key": "zh:yi-ge:phrase"}
    brief = build_agent_brief(db)
    assert "zh:yi-ge:phrase" not in {term.key for term in brief.priority_terms}
    assert brief.recent_lessons[0].target_term_keys == []


def test_composite_reveal_distributes_one_failure_by_current_component_state(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    laugh = ("zh:laugh", "笑", "笑", "verb", "laugh", 600)
    completed = ("zh:le:completed", "了", "了", "particle", "completed action", 5)
    changed = ("zh:le:changed", "了", "了", "particle", "change of state", 5)
    composite = ("zh:laughed", "笑了", "笑了", "verb", "laughed", 600)
    laugh_lesson = import_lesson(
        db, lesson_factory(key="laugh-component", learning_language="zh-Hans", terms=[laugh])
    )
    completed_lesson = import_lesson(
        db,
        lesson_factory(
            key="completed-component",
            learning_language="zh-Hans",
            terms=[completed],
        ),
    )
    import_lesson(
        db,
        lesson_factory(
            key="changed-component",
            learning_language="zh-Hans",
            terms=[changed],
        ),
    )
    composite_lesson = import_lesson(
        db,
        lesson_factory(
            key="composite-laugh",
            learning_language="zh-Hans",
            terms=[composite],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                laugh_lesson.id,
                "term.revealed",
                event_id="weaken-laugh",
                session_id="laugh-setup",
                payload={"term_key": laugh[0]},
                seconds=1,
            ),
            event_factory(
                completed_lesson.id,
                "lesson.completed",
                event_id="establish-completed-le",
                session_id="le-setup",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=45,
            ),
        ],
    )
    before = {key: state.reveal_failures for key, state in _states(db).items()}
    record_events(
        db,
        [
            event_factory(
                composite_lesson.id,
                "term.revealed",
                event_id="reveal-laughed",
                session_id="composite-session",
                payload={"term_key": composite[0]},
                seconds=100,
            ),
            event_factory(
                composite_lesson.id,
                "term.revealed",
                event_id="repeat-laughed",
                session_id="composite-session",
                payload={"term_key": composite[0]},
                seconds=101,
            ),
            event_factory(
                composite_lesson.id,
                "lesson.completed",
                event_id="complete-composite",
                session_id="composite-session",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=145,
            ),
        ],
    )

    after = _states(db)
    laugh_weight = after[laugh[0]].reveal_failures - before[laugh[0]]
    le_weight = after[completed[0]].reveal_failures - before[completed[0]]
    assert 0 < le_weight < laugh_weight < 1
    assert laugh_weight + le_weight == pytest.approx(1)
    assert after[changed[0]].reveal_failures == 0
    assert after[laugh[0]].qualified_exposures == 0
    assert after[completed[0]].qualified_exposures == 1
    assert (
        db.scalar(
            select(func.count())
            .select_from(Interaction)
            .where(
                Interaction.event_id.in_(["reveal-laughed", "repeat-laughed"]),
            )
        )
        == 2
    )


def test_exact_definition_aliases_share_one_srs_identity(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    canonical = ("es:ir:canonical", "voy", "ir", "VERB", "to go", 25)
    alias = ("es:ir:alias", "fuimos", "ir", "verb", "to go", 999)
    import_lesson(
        db,
        lesson_factory(key="canonical-ir", terms=[canonical], targets=[]),
    )
    alias_lesson = import_lesson(
        db,
        lesson_factory(key="alias-ir", terms=[alias], targets=[alias[0]]),
    )

    assert set(_states(db)) == {canonical[0]}
    assert get_reader_state(db, alias_lesson.id).term_bands[alias[0]] == "focus"
    record_events(
        db,
        [
            event_factory(
                alias_lesson.id,
                "term.revealed",
                event_id="reveal-ir-alias",
                payload={"term_key": alias[0]},
            )
        ],
    )
    states = _states(db)
    assert set(states) == {canonical[0]}
    assert states[canonical[0]].reveal_failures == 1


def test_cross_session_reveals_surface_common_surprises_without_promoting_rare_incidentals(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    common = ("es:comun:WORD", "común", "común", "WORD", "common", 10)
    rare_incidental = ("es:raro:WORD", "raro", "raro", "WORD", "rare", 100_000)
    rare_target = ("es:meta:WORD", "meta", "meta", "WORD", "target", 90_000)
    known_lesson = import_lesson(
        db,
        lesson_factory(key="known-common", terms=[common], targets=[]),
    )
    evidence_lesson = import_lesson(
        db,
        lesson_factory(
            key="revealed-vocabulary",
            terms=[common, rare_incidental, rare_target],
            targets=[rare_target[0]],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                known_lesson.id,
                "lesson.completed",
                event_id="known-common-completion",
                session_id="known-session",
                payload={"active_seconds": 45, "completion_ratio": 1},
            ),
            event_factory(
                evidence_lesson.id,
                "term.revealed",
                event_id="common-first-click",
                session_id="reveal-session-one",
                payload={"term_key": common[0]},
                seconds=10,
            ),
            event_factory(
                evidence_lesson.id,
                "term.revealed",
                event_id="common-repeat-same-session",
                session_id="reveal-session-one",
                payload={"term_key": common[0]},
                seconds=11,
            ),
            event_factory(
                evidence_lesson.id,
                "term.revealed",
                event_id="rare-incidental-click",
                session_id="reveal-session-one",
                payload={"term_key": rare_incidental[0]},
                seconds=12,
            ),
            event_factory(
                evidence_lesson.id,
                "term.revealed",
                event_id="rare-target-click",
                session_id="reveal-session-one",
                payload={"term_key": rare_target[0]},
                seconds=13,
            ),
            event_factory(
                evidence_lesson.id,
                "term.revealed",
                event_id="common-second-session",
                session_id="reveal-session-two",
                payload={"term_key": common[0]},
                seconds=20,
            ),
        ],
    )

    states = _states(db)
    assert states[common[0]].qualified_exposures == 1
    assert states[common[0]].reveal_failures == 2
    assert states[rare_incidental[0]].reveal_failures == 1
    assert states[rare_target[0]].reveal_failures == 1

    priority = {term.key: term for term in build_agent_brief(db).priority_terms}
    assert priority[common[0]].urgency > priority[rare_incidental[0]].urgency
    assert priority[rare_target[0]].urgency > priority[rare_incidental[0]].urgency


def test_target_policy_scales_with_length_confidence_and_profile(
    db: Session, lesson_factory: Any
) -> None:
    terms = [
        (f"es:pool-{index}:NOUN", f"pool{index}", f"pool{index}", "NOUN", "pool", index)
        for index in range(1, 81)
    ]
    import_lesson(db, lesson_factory(key="candidate-pool", terms=terms, targets=[]))
    update_profile(
        db,
        {"preferences": {"text_length": 360, "target_known_ratio": 0.95}},
    )

    uncertain_brief = build_agent_brief(db)
    uncertain = build_generation_target_policy(db, "lesson", brief=uncertain_brief)

    state = db.get(ProficiencyState, 1)
    assert state is not None
    state.status = "stable"
    state.lower = 0.48
    state.upper = 0.52
    db.commit()
    certain_brief = build_agent_brief(db)
    certain = build_generation_target_policy(db, "lesson", brief=certain_brief)

    assert certain.max_count > uncertain.max_count
    assert len(certain.candidate_term_keys) < len(uncertain.candidate_term_keys)

    update_profile(
        db,
        {
            "difficulty": 0.9,
            "preferences": {"text_length": 720, "target_known_ratio": 0.5},
        },
    )
    expanded = build_generation_target_policy(db, "lesson")
    assert expanded.target_text_length == 720
    assert expanded.preferred_count > certain.preferred_count
    assert len(expanded.candidate_term_keys) > len(certain.candidate_term_keys)
    assert expanded.max_count <= len(expanded.candidate_term_keys)


def test_target_policy_uses_feedback_availability_and_probe_load(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory(targets=[]))
    calibration = build_generation_target_policy(db, "calibration")
    assert calibration.target_text_length == 192
    assert calibration.preferred_count == calibration.max_count == 0
    assert calibration.candidate_term_keys == []

    update_profile(
        db,
        {
            "level": "B1",
            "preferences": {"text_length": 360, "target_known_ratio": 0.8},
        },
    )
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="shorter-policy-completion",
                session_id="shorter-session",
                payload={"active_seconds": 60, "completion_ratio": 1},
            ),
            event_factory(
                lesson.id,
                "lesson.rated",
                event_id="shorter-policy",
                session_id="shorter-session",
                payload={"rating": 1, "feedback": ["shorter"]},
                seconds=1,
            ),
        ],
    )
    shorter = build_generation_target_policy(db, "lesson")
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="longer-policy-completion",
                session_id="longer-session",
                payload={"active_seconds": 60, "completion_ratio": 1},
                seconds=2,
            ),
            event_factory(
                lesson.id,
                "lesson.rated",
                event_id="longer-policy",
                session_id="longer-session",
                payload={"rating": 1, "feedback": ["longer"]},
                seconds=3,
            ),
        ],
    )
    longer = build_generation_target_policy(db, "lesson")

    assert shorter.target_text_length == 270
    assert longer.target_text_length == 480
    assert longer.preferred_count > shorter.preferred_count
    assert longer.max_count <= len(longer.candidate_term_keys) == 3


def test_recent_skipped_offer_rotates_without_changing_urgency(
    db: Session, lesson_factory: Any
) -> None:
    terms = [
        (f"es:rotate-{index}:NOUN", f"rotate{index}", f"rotate{index}", "NOUN", "term", index)
        for index in range(1, 6)
    ]
    import_lesson(db, lesson_factory(key="rotation-base", terms=terms, targets=[]))
    before = build_agent_brief(db).priority_terms
    cooled_key = before[0].key
    before_urgency = before[0].urgency

    import_lesson(
        db,
        lesson_factory(
            key="rotation-offer",
            terms=terms,
            targets=[],
            metadata={
                "generation_mode": "lesson",
                "targets": {"offered": [cooled_key], "realized": []},
            },
        ),
    )
    rotated = build_agent_brief(db).priority_terms

    assert rotated[-1].key == cooled_key
    assert rotated[-1].urgency == before_urgency


def test_full_translation_excludes_all_passive_evidence(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="translate-all",
                payload={"scope": "lesson"},
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="complete",
                payload={"active_seconds": 60, "completion_ratio": 1},
                seconds=60,
            ),
        ],
    )
    assert all(state.qualified_exposures == 0 for state in _states(db).values())


def test_reader_advances_to_next_uncompleted_lesson(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    first = import_lesson(db, lesson_factory(key="first"))
    second = import_lesson(db, lesson_factory(key="second"))

    assert get_reader_state(db).lesson_id == first.id
    record_events(
        db,
        [
            event_factory(
                first.id,
                "lesson.completed",
                event_id="complete-first",
                payload={"active_seconds": 60, "completion_ratio": 1},
            )
        ],
    )
    assert get_reader_state(db).lesson_id == second.id


def test_reader_term_bands_combine_evidence_frequency_and_targets(
    db: Session, lesson_factory: Any, event_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Knowledge is time-aware: evaluate just after spaced readings over a month.
    monkeypatch.setattr(
        "server.models.utc_now", lambda: datetime(2026, 2, 2, 12, 0, tzinfo=timezone.utc)
    )
    familiar = ("es:familiar:WORD", "familiar", "familiar", "WORD", "familiar", 90_000)
    prior_lessons = [
        import_lesson(
            db,
            lesson_factory(key=f"familiar-{index}", terms=[familiar], targets=[]),
        )
        for index in range(5)
    ]
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id=f"familiar-complete-{index}",
                session_id=f"familiar-session-{index}",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=day * 86_400,
            )
            for index, (lesson, day) in enumerate(
                zip(prior_lessons, (0, 3, 8, 16, 31), strict=True)
            )
        ],
    )

    expected = ("es:expected:WORD", "expected", "expected", "WORD", "expected", 100)
    uncertain = ("es:uncertain:WORD", "uncertain", "uncertain", "WORD", "uncertain", 5_000)
    rare = ("es:rare:WORD", "rare", "rare", "WORD", "rare", 100_000)
    deliberate = ("es:focus:WORD", "focus", "focus", "WORD", "focus", 100_000)
    reader_lesson = import_lesson(
        db,
        lesson_factory(
            key="reader-bands",
            terms=[familiar, expected, uncertain, rare, deliberate],
            targets=[familiar[0], deliberate[0]],
        ),
    )

    reader = get_reader_state(db, lesson_id=reader_lesson.id)

    assert reader.term_bands == {
        familiar[0]: "familiar",
        expected[0]: "expected",
        uncertain[0]: "uncertain",
        rare[0]: "incidental",
        deliberate[0]: "focus",
    }


def test_reader_term_bands_treat_one_reveal_as_a_check_but_repeated_reveals_as_signal(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    common = ("es:common-check:WORD", "common", "common", "WORD", "common", 100)
    lesson = import_lesson(
        db,
        lesson_factory(key="reader-checks", terms=[common], targets=[]),
    )
    assert get_reader_state(db, lesson_id=lesson.id).term_bands[common[0]] == "expected"

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="reader-first-check",
                session_id="reader-first-check",
                payload={"term_key": common[0]},
            )
        ],
    )
    assert get_reader_state(db, lesson_id=lesson.id).term_bands[common[0]] == "expected"

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="reader-second-check",
                session_id="reader-second-check",
                payload={"term_key": common[0]},
                seconds=1,
            )
        ],
    )
    assert get_reader_state(db, lesson_id=lesson.id).term_bands[common[0]] == "uncertain"

    focus_lesson = import_lesson(
        db,
        lesson_factory(key="reader-checks-focus", terms=[common], targets=[common[0]]),
    )
    assert get_reader_state(db, lesson_id=focus_lesson.id).term_bands[common[0]] == "focus"


def test_reader_term_bands_use_profile_and_conservative_proficiency(
    db: Session, lesson_factory: Any
) -> None:
    term = ("es:ranked:WORD", "ranked", "ranked", "WORD", "ranked", 2_500)
    lesson = import_lesson(db, lesson_factory(key="reader-level", terms=[term], targets=[]))

    update_profile(db, {"difficulty": 0.1})
    assert get_reader_state(db, lesson_id=lesson.id).term_bands[term[0]] == "incidental"

    update_profile(db, {"difficulty": 0.8})
    assert get_reader_state(db, lesson_id=lesson.id).term_bands[term[0]] == "expected"

    proficiency = db.get(ProficiencyState, 1)
    assert proficiency is not None
    proficiency.status = "stable"
    proficiency.estimate = 0.15
    proficiency.lower = 0.1
    proficiency.upper = 0.2
    db.commit()
    assert get_reader_state(db, lesson_id=lesson.id).term_bands[term[0]] == "incidental"


def test_reader_term_bands_treat_unranked_proper_names_as_context(
    db: Session, lesson_factory: Any
) -> None:
    name = ("zh:xiaolin:PROPN", "小林", "小林", "PROPN", "Xiaolin", None)
    update_profile(db, {"learning_language": "zh-Hans"})
    lesson = import_lesson(
        db,
        lesson_factory(
            key="reader-name",
            learning_language="zh-Hans",
            terms=[name],
            targets=[],
        ),
    )

    assert get_reader_state(db, lesson_id=lesson.id).term_bands[name[0]] == "incidental"


def test_reader_term_band_contract_rejects_unknown_categories(
    db: Session, lesson_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory(targets=[]))
    payload = get_reader_state(db, lesson_id=lesson.id).model_dump(mode="json")
    payload["term_bands"]["es:manana:NOUN"] = "maybe-ish"

    with pytest.raises(
        ValidationError,
        match="familiar|expected|uncertain|focus|incidental",
    ):
        ReaderState.model_validate(payload)


@pytest.mark.parametrize(
    "profile_patch",
    [{"learning_language": "fr"}, {"translation_language": "de"}],
)
def test_explicit_reads_and_events_reject_inactive_language_lessons(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
    profile_patch: dict[str, str],
) -> None:
    lesson = import_lesson(db, lesson_factory())
    update_profile(db, profile_patch)

    with pytest.raises(LookupError, match="lesson not found"):
        get_reader_state(db, lesson_id=lesson.id)
    with pytest.raises(LookupError, match="lessons not found"):
        record_events(
            db,
            [
                event_factory(
                    lesson.id,
                    "lesson.started",
                    event_id=f"inactive-language-{next(iter(profile_patch))}",
                )
            ],
        )
    assert db.scalar(select(func.count()).select_from(Interaction)) == 0


def test_rebuild_produces_the_same_learning_state(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="reveal",
                payload={"term_key": "es:manana:NOUN"},
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="complete",
                payload={"active_seconds": 60, "completion_ratio": 1},
                seconds=60,
            ),
        ],
    )
    before = _state_snapshot(db)
    rebuild_lexeme_states(db)
    assert _state_snapshot(db) == before


def test_agent_brief_isolates_language_and_preserves_targets(
    db: Session, lesson_factory: Any
) -> None:
    import_lesson(db, lesson_factory(key="spanish"))
    update_profile(
        db,
        {
            "learning_language": "fr",
            "translation_language": "en",
            "interests": ["books"],
        },
    )
    french_terms = [
        ("shared:greeting", "bonjour", "bonjour", "INTJ", "hello", 1),
        ("fr:livre:NOUN", "livre", "livre", "NOUN", "book", 100),
    ]
    french = lesson_factory(
        key="french",
        learning_language="fr",
        terms=french_terms,
        targets=["fr:livre:NOUN", "shared:greeting"],
        metadata={
            "series": "library",
            "content_plan": {"archetype": "mini-interview"},
            "targets": {"offered": ["stale:term"]},
            "grammar": {"offered": ["stale:construction"]},
        },
    )
    import_lesson(db, french)

    brief = build_agent_brief(db)

    assert brief.profile.learning_language == "fr"
    assert [term.key for term in brief.priority_terms] == [
        "shared:greeting",
        "fr:livre:NOUN",
    ]
    assert brief.lesson_count == 1
    assert brief.recent_lessons[0].target_term_keys == [
        "fr:livre:NOUN",
        "shared:greeting",
    ]
    recent = brief.recent_lessons[0]
    assert recent.metadata == {
        "series": "library",
        "content_plan": {"archetype": "mini-interview"},
    }
    assert recent.opening_excerpt.startswith("<script>")
    assert recent.ending_excerpt.endswith("livre.")
    assert recent.calibration_lesson is False
