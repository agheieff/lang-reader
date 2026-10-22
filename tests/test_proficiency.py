from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from server import profile_state
from server.agent_brief import build_agent_brief, calibration_generation_brief
from server.agent_worker import _validate_generated_calibration
from server.calibration import INITIAL_PROBE_DIFFICULTY
from server.generation_queue import ensure_generation_task, unread_generation_lesson_count
from server.learning import import_lesson, record_events, validate_lesson
from server.models import Base, ProficiencyState
from server.profile_fingerprint import profile_fingerprint
from server.profile_state import (
    PROFILE_LEVEL_SET_AT_KEY,
    build_profile_view,
    ensure_profile,
    profile_level_source,
    rebuild_proficiency_state,
    update_profile,
)
from server.schemas import CalibrationGenerationBrief, LessonCalibration
from server.statistics import get_statistics_summary


def _calibration_lesson(
    lesson_factory: Any,
    *,
    key: str,
    prefix: str,
    sequence: int,
    difficulties: list[float] | None = None,
    learning_language: str = "es-ES",
) -> dict[str, Any]:
    values = difficulties or [round(0.10 + index * 0.06, 6) for index in range(12)]
    terms = [
        (
            f"{prefix}:term-{index}:NOUN",
            f"palabra{index}",
            f"palabra{index}",
            "NOUN",
            f"word {index}",
            index + 1,
        )
        for index in range(len(values))
    ]
    payload = cast(
        dict[str, Any],
        lesson_factory(
            key=key,
            learning_language=learning_language,
            terms=terms,
            targets=[],
            metadata={"purpose": "calibration"},
        ),
    )
    payload["calibration"] = {
        "sequence": sequence,
        "probes": [
            {"term_key": term[0], "difficulty": difficulty}
            for term, difficulty in zip(terms, values, strict=True)
        ],
    }
    return payload


def _reading_events(
    event_factory: Any,
    lesson_id: int,
    term_keys: list[str],
    *,
    event_prefix: str,
    revealed: set[str] | None = None,
    active_seconds: float = 60,
    completion_ratio: float = 1,
    translated_sentences: tuple[int, ...] = (),
    full_translation: bool = False,
    translation_after_completion: bool = False,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, term_key in enumerate(term_keys):
        if revealed is not None and term_key in revealed:
            events.append(
                event_factory(
                    lesson_id,
                    "term.revealed",
                    event_id=f"{event_prefix}-reveal-{index}",
                    payload={"term_key": term_key},
                    seconds=index + 1,
                )
            )
    translation_second = 61 if translation_after_completion else 20
    for index in translated_sentences:
        events.append(
            event_factory(
                lesson_id,
                "translation.revealed",
                event_id=f"{event_prefix}-sentence-{index}",
                payload={"scope": "sentence", "sentence_key": f"sentence-{index + 1}"},
                seconds=translation_second,
            )
        )
    if full_translation:
        events.append(
            event_factory(
                lesson_id,
                "translation.revealed",
                event_id=f"{event_prefix}-full",
                payload={"scope": "lesson"},
                seconds=translation_second,
            )
        )
    events.append(
        event_factory(
            lesson_id,
            "lesson.completed",
            event_id=f"{event_prefix}-complete",
            payload={
                "active_seconds": active_seconds,
                "completion_ratio": completion_ratio,
            },
            seconds=60,
        )
    )
    return events


def test_calibration_schema_requires_unique_known_single_occurrence_probes(
    lesson_factory: Any,
) -> None:
    payload = _calibration_lesson(
        lesson_factory,
        key="calibration-schema",
        prefix="schema",
        sequence=1,
    )
    assert validate_lesson(payload).calibration is not None

    too_few = deepcopy(payload)
    too_few["calibration"]["probes"] = too_few["calibration"]["probes"][:7]
    with pytest.raises(ValidationError, match="at least 8"):
        validate_lesson(too_few)

    duplicate = deepcopy(payload)
    duplicate["calibration"]["probes"][1] = duplicate["calibration"]["probes"][0]
    with pytest.raises(ValidationError, match="unique"):
        validate_lesson(duplicate)

    unknown = deepcopy(payload)
    unknown["calibration"]["probes"][0]["term_key"] = "missing"
    with pytest.raises(ValidationError, match="unknown probe"):
        validate_lesson(unknown)

    repeated = deepcopy(payload)
    repeated_run = deepcopy(repeated["blocks"][0]["sentences"][0]["runs"][1])
    repeated["blocks"][0]["sentences"][1]["runs"].insert(0, repeated_run)
    with pytest.raises(ValidationError, match="exactly once"):
        validate_lesson(repeated)


def test_unknown_profile_queues_one_calibration_despite_an_ordinary_unread_lesson(
    db: Session, lesson_factory: Any
) -> None:
    profile = ensure_profile(db)
    import_lesson(db, lesson_factory(key="ordinary-unread"))

    task = ensure_generation_task(db, require_enabled=False)
    request = calibration_generation_brief(db)

    assert profile_level_source(profile) == "unknown"
    assert profile.level == "B1"
    assert profile.difficulty == INITIAL_PROBE_DIFFICULTY
    assert unread_generation_lesson_count(db, "lesson") == 1
    assert unread_generation_lesson_count(db, "calibration") == 0
    assert task is not None
    assert task.payload["generation_mode"] == "calibration"
    assert task.payload["queue_target"] == 1
    assert task.payload["needed_lesson_count"] == 1
    assert request is not None
    assert request.sequence == 1
    assert request.target_difficulty == pytest.approx(INITIAL_PROBE_DIFFICULTY)
    assert len(request.probe_difficulties) == 12


def test_calibration_fingerprint_changes_when_excluded_probes_change(
    db: Session, lesson_factory: Any
) -> None:
    before = profile_fingerprint(db)
    import_lesson(
        db,
        _calibration_lesson(
            lesson_factory,
            key="fingerprint-calibration",
            prefix="fingerprint",
            sequence=1,
        ),
    )

    assert profile_fingerprint(db) != before


def test_two_qualified_lessons_adapt_and_estimate_unknown_profile(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    first_payload = _calibration_lesson(
        lesson_factory,
        key="calibration-one",
        prefix="first",
        sequence=1,
    )
    first = import_lesson(db, first_payload)
    first_probes = first_payload["calibration"]["probes"]
    first_revealed = {probe["term_key"] for probe in first_probes if probe["difficulty"] > 0.38}
    record_events(
        db,
        _reading_events(
            event_factory,
            first.id,
            [probe["term_key"] for probe in first_probes],
            event_prefix="first",
            revealed=first_revealed,
        ),
    )

    after_first = db.get(ProficiencyState, 1)
    second_request = calibration_generation_brief(db)
    assert after_first is not None
    assert after_first.status == "collecting"
    assert after_first.qualified_attempts == 1
    assert profile_level_source(ensure_profile(db)) == "unknown"
    assert second_request is not None
    assert second_request.sequence == 2
    assert second_request.target_difficulty == pytest.approx(after_first.estimate)
    assert set(second_request.excluded_term_keys) == {probe["term_key"] for probe in first_probes}

    second_payload = _calibration_lesson(
        lesson_factory,
        key="calibration-two",
        prefix="second",
        sequence=2,
        difficulties=second_request.probe_difficulties,
    )
    second = import_lesson(db, second_payload)
    second_probes = second_payload["calibration"]["probes"]
    second_revealed = {probe["term_key"] for probe in second_probes if probe["difficulty"] > 0.38}
    record_events(
        db,
        _reading_events(
            event_factory,
            second.id,
            [probe["term_key"] for probe in second_probes],
            event_prefix="second",
            revealed=second_revealed,
        ),
    )

    state = db.get(ProficiencyState, 1)
    profile = ensure_profile(db)
    brief = build_agent_brief(db)
    assert state is not None
    assert state.status == "rough"
    assert state.qualified_attempts == 2
    assert state.usable_probes == 24
    assert profile_level_source(profile) == "estimated"
    assert brief.profile.difficulty == state.estimate
    assert brief.profile.level == state.level
    assert brief.profile.proficiency.status == "rough"
    assert brief.profile.proficiency.lower is not None
    assert brief.profile.proficiency.upper is not None

    stored_prior = profile.difficulty
    baseline = state.estimate
    assert baseline is not None
    fingerprint = profile_fingerprint(db)
    ordinary_payload = lesson_factory(
        key="ordinary-adaptation",
        metadata={"adaptive_target_difficulty": baseline},
    )
    ordinary_payload["difficulty"] = baseline
    ordinary = import_lesson(db, ordinary_payload)
    record_events(
        db,
        _reading_events(
            event_factory,
            ordinary.id,
            [],
            event_prefix="ordinary-adaptation",
        ),
    )

    adapted_profile = ensure_profile(db)
    adapted_state = db.get(ProficiencyState, 1)
    assert adapted_state is not None
    assert build_profile_view(db).difficulty > baseline
    assert adapted_profile.difficulty == stored_prior
    assert adapted_state.qualified_attempts == 2
    assert adapted_state.usable_probes == 24
    assert profile_fingerprint(db) == fingerprint


def test_self_reported_level_is_never_overwritten_and_demo_metadata_is_ignored(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    update_profile(db, {"level": "A2", "difficulty": 0.27})
    demo = import_lesson(
        db,
        lesson_factory(
            key="ui-demo",
            metadata={"purpose": "interactive test lesson"},
        ),
    )
    record_events(
        db,
        _reading_events(
            event_factory,
            demo.id,
            [],
            event_prefix="demo",
            active_seconds=65,
            full_translation=True,
        ),
    )
    assert rebuild_proficiency_state(db).status == "collecting"

    for sequence in (1, 2):
        payload = _calibration_lesson(
            lesson_factory,
            key=f"self-calibration-{sequence}",
            prefix=f"self-{sequence}",
            sequence=sequence,
        )
        lesson = import_lesson(db, payload)
        probes = payload["calibration"]["probes"]
        record_events(
            db,
            _reading_events(
                event_factory,
                lesson.id,
                [probe["term_key"] for probe in probes],
                event_prefix=f"self-{sequence}",
                revealed={probe["term_key"] for probe in probes},
            ),
        )

    profile = ensure_profile(db)
    state = db.get(ProficiencyState, 1)
    assert state is not None
    assert state.status == "rough"
    assert profile_level_source(profile) == "self_reported"
    assert profile.level == "A2"
    assert profile.difficulty == 0.27


def test_self_reported_level_has_one_post_anchor_adaptive_snapshot(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    old_payload = lesson_factory(key="old-reading")
    old_payload["difficulty"] = 0.20
    old = import_lesson(db, old_payload)
    old_events = _reading_events(
        event_factory,
        old.id,
        [],
        event_prefix="old-reading",
    )
    for event in old_events:
        event["occurred_at"] = datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat()
    record_events(db, old_events)

    update_profile(db, {"level": "A2", "difficulty": 0.30})
    configured = ensure_profile(db)
    assert build_agent_brief(db).profile.difficulty == pytest.approx(0.30)
    stable_fingerprint = profile_fingerprint(db)

    payload = lesson_factory(
        key="post-anchor-reading",
        metadata={"adaptive_target_difficulty": 0.30},
    )
    payload["difficulty"] = 0.30
    lesson = import_lesson(db, payload)
    future = datetime.now(timezone.utc) + timedelta(days=1)
    first_session = [
        event_factory(
            lesson.id,
            "lesson.completed",
            event_id="post-anchor-complete",
            session_id="post-anchor",
            payload={"active_seconds": 90, "completion_ratio": 1},
        ),
        event_factory(
            lesson.id,
            "lesson.rated",
            event_id="post-anchor-rating",
            session_id="post-anchor",
            payload={"feedback": ["more_challenging"]},
            seconds=1,
        ),
    ]
    for index, event in enumerate(first_session):
        event["occurred_at"] = (future + timedelta(seconds=index)).isoformat()
    record_events(db, first_session)

    state = db.get(ProficiencyState, 1)
    assert state is not None
    view = build_profile_view(db)
    effective = view.difficulty
    assert state.estimate == pytest.approx(effective)
    assert build_agent_brief(db).profile.difficulty == pytest.approx(effective)
    assert get_statistics_summary(db).level.value == pytest.approx(effective)
    assert ensure_profile(db).difficulty == pytest.approx(configured.difficulty)
    assert effective > configured.difficulty
    assert profile_fingerprint(db) == stable_fingerprint

    reread = [
        event_factory(
            lesson.id,
            "lesson.completed",
            event_id="reread-complete",
            session_id="reread",
            payload={"active_seconds": 90, "completion_ratio": 1},
        ),
        event_factory(
            lesson.id,
            "lesson.rated",
            event_id="reread-rating",
            session_id="reread",
            payload={"feedback": ["easier"]},
            seconds=1,
        ),
    ]
    for index, event in enumerate(reread):
        event["occurred_at"] = (future + timedelta(days=1, seconds=index)).isoformat()
    record_events(db, reread)

    assert build_profile_view(db).difficulty == pytest.approx(effective)
    assert build_agent_brief(db).profile.difficulty == pytest.approx(effective)
    assert get_statistics_summary(db).level.value == pytest.approx(effective)
    assert ensure_profile(db).difficulty == pytest.approx(0.30)


def test_migrated_self_reported_profile_gets_one_stable_level_anchor(db: Session) -> None:
    profile = ensure_profile(db)
    original_anchor = datetime(2025, 4, 3, tzinfo=timezone.utc)
    profile.preferences = {"level_source": "self_reported"}
    profile.updated_at = original_anchor
    db.commit()

    migrated = ensure_profile(db)
    stored_anchor = migrated.preferences[PROFILE_LEVEL_SET_AT_KEY]
    assert stored_anchor == original_anchor.isoformat()

    update_profile(db, {"preferences": {"theme": "dark"}})
    assert ensure_profile(db).preferences[PROFILE_LEVEL_SET_AT_KEY] == stored_anchor


def test_qualification_filters_and_completion_cutoff_integrate(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    invalid_specs: tuple[tuple[str, dict[str, Any]], ...] = (
        ("short", {"active_seconds": 2}),
        ("incomplete", {"completion_ratio": 0.7}),
        ("full", {"full_translation": True}),
    )
    for sequence, (name, overrides) in enumerate(invalid_specs, 1):
        payload = _calibration_lesson(
            lesson_factory,
            key=f"invalid-{name}",
            prefix=name,
            sequence=sequence,
        )
        lesson = import_lesson(db, payload)
        probes = payload["calibration"]["probes"]
        record_events(
            db,
            _reading_events(
                event_factory,
                lesson.id,
                [probe["term_key"] for probe in probes],
                event_prefix=name,
                **overrides,
            ),
        )
    assert rebuild_proficiency_state(db).status == "unstarted"

    values = [round(0.15 + index * 0.06, 6) for index in range(10)]
    partial_payload = _calibration_lesson(
        lesson_factory,
        key="partial-translation",
        prefix="partial",
        sequence=1,
        difficulties=values,
    )
    partial = import_lesson(db, partial_payload)
    probes = partial_payload["calibration"]["probes"]
    record_events(
        db,
        _reading_events(
            event_factory,
            partial.id,
            [probe["term_key"] for probe in probes],
            event_prefix="partial",
            translated_sentences=(0, 1),
            full_translation=True,
            translation_after_completion=True,
        )
        + [
            event_factory(
                partial.id,
                "lesson.completed",
                event_id="partial-second-completion",
                payload={"active_seconds": 2, "completion_ratio": 1},
                seconds=70,
            )
        ],
    )

    state = db.get(ProficiencyState, 1)
    assert state is not None
    assert state.status == "collecting"
    assert state.qualified_attempts == 1
    assert state.usable_probes == 10


def test_duplicate_event_replay_recovers_failed_derived_rebuild(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _calibration_lesson(
        lesson_factory,
        key="rebuild-recovery",
        prefix="recovery",
        sequence=1,
    )
    lesson = import_lesson(db, payload)
    probes = payload["calibration"]["probes"]
    events = _reading_events(
        event_factory,
        lesson.id,
        [probe["term_key"] for probe in probes],
        event_prefix="recovery",
    )
    original = profile_state.rebuild_proficiency_state

    def fail_rebuild(_: Session) -> ProficiencyState:
        raise RuntimeError("derived rebuild failed")

    monkeypatch.setattr(profile_state, "rebuild_proficiency_state", fail_rebuild)
    with pytest.raises(RuntimeError, match="derived rebuild failed"):
        record_events(db, events)
    monkeypatch.setattr(profile_state, "rebuild_proficiency_state", original)

    replay = record_events(db, events)
    state = db.get(ProficiencyState, 1)
    assert replay.accepted == 0
    assert replay.duplicates == len(events)
    assert state is not None
    assert state.qualified_attempts == 1


def test_host_rejects_calibration_output_that_does_not_match_request() -> None:
    requested = CalibrationGenerationBrief(
        sequence=2,
        target_difficulty=0.45,
        probe_difficulties=[0.10 + index * 0.05 for index in range(8)],
        excluded_term_keys=["old"],
    )
    generated = LessonCalibration(
        sequence=2,
        probes=[
            {"term_key": f"new-{index}", "difficulty": difficulty}
            for index, difficulty in enumerate(requested.probe_difficulties)
        ],
    )
    _validate_generated_calibration(generated, "calibration", requested)

    wrong_sequence = generated.model_copy(update={"sequence": 1})
    with pytest.raises(ValueError, match="wrong calibration sequence"):
        _validate_generated_calibration(wrong_sequence, "calibration", requested)
    reused = generated.model_copy(
        update={
            "probes": [
                generated.probes[0].model_copy(update={"term_key": "old"}),
                *generated.probes[1:],
            ]
        }
    )
    with pytest.raises(ValueError, match="excluded"):
        _validate_generated_calibration(reused, "calibration", requested)
    with pytest.raises(ValueError, match="must not return"):
        _validate_generated_calibration(generated, "lesson", None)


def test_proficiency_is_isolated_between_language_databases(
    tmp_path: Path, lesson_factory: Any, event_factory: Any
) -> None:
    estimates: list[float | None] = []
    for name, language, reveal_all in (
        ("spanish", "es-ES", True),
        ("chinese", "zh-Hant", False),
    ):
        engine = create_engine(f"sqlite:///{tmp_path / f'{name}.db'}")
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as db:
            ensure_profile(db, learning_language=language)
            payload = _calibration_lesson(
                lesson_factory,
                key=f"{name}-calibration",
                prefix=name,
                sequence=1,
                learning_language=language,
            )
            lesson = import_lesson(db, payload)
            probes = payload["calibration"]["probes"]
            record_events(
                db,
                _reading_events(
                    event_factory,
                    lesson.id,
                    [probe["term_key"] for probe in probes],
                    event_prefix=name,
                    revealed=({probe["term_key"] for probe in probes} if reveal_all else set()),
                ),
            )
            state = db.get(ProficiencyState, 1)
            assert state is not None
            assert state.qualified_attempts == 1
            estimates.append(state.estimate)
        engine.dispose()

    assert estimates[0] is not None
    assert estimates[1] is not None
    assert estimates[0] < estimates[1]
