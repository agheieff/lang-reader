from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from server.agent_brief import build_agent_brief
from server.generation_queue import (
    claim_generation_task,
    ensure_generation_task,
    fail_generation_task,
    maintain_generation_task,
)
from server.grammar import get_grammar_state
from server.learning import LessonConflictError, get_reader_state, import_lesson, record_events
from server.lesson_queue import (
    LessonQueueConflictError,
    record_lesson_queue_action,
    record_lesson_queue_move,
    unread_lessons,
)
from server.models import (
    GenerationTask,
    Interaction,
    LessonQueueAction,
    LessonQueueMoveAction,
    LexemeState,
)
from server.profile_state import ensure_profile, update_profile
from server.texts import get_texts_state
from server.words import get_words_state
from server.workspaces import Workspace


def _with_grammar(payload: dict[str, Any]) -> dict[str, Any]:
    payload["blocks"][0]["sentences"][0]["grammar"] = [
        {
            "key": f"{payload['key']}:present",
            "construction_key": "es:present-indicative",
            "run_start": 1,
            "run_end": 2,
        }
    ]
    return payload


def test_skip_advances_reader_and_restore_recovers_queue_without_evidence(
    db: Session, lesson_factory: Any
) -> None:
    first = import_lesson(db, _with_grammar(lesson_factory(key="first")))
    second = import_lesson(db, lesson_factory(key="second"))
    before = [
        (state.term_key, state.alpha, state.beta, state.qualified_exposures)
        for state in db.scalars(select(LexemeState).order_by(LexemeState.term_key)).all()
    ]

    skipped = record_lesson_queue_action(db, lesson_id=first.id, action_id=uuid4(), skipped=True)

    texts = {item.id: item for item in get_texts_state(db).texts}
    assert skipped.skipped is True
    assert texts[first.id].status == "skipped"
    assert texts[first.id].skipped_at == skipped.occurred_at
    assert texts[first.id].queue_position is None
    assert texts[first.id].opened_at is None
    assert texts[first.id].session_count == 0
    assert texts[second.id].status == "queued"
    assert texts[second.id].queue_position == 1
    assert get_reader_state(db).lesson_id == second.id
    assert get_reader_state(db, lesson_id=first.id).lesson_id == first.id
    assert len(unread_lessons(db)) == 1
    assert db.scalar(select(func.count()).select_from(Interaction)) == 0
    assert get_words_state(db).words == []
    assert get_grammar_state(db).constructions == []
    after = [
        (state.term_key, state.alpha, state.beta, state.qualified_exposures)
        for state in db.scalars(select(LexemeState).order_by(LexemeState.term_key)).all()
    ]
    assert after == before

    restored = record_lesson_queue_action(db, lesson_id=first.id, action_id=uuid4(), skipped=False)
    texts = {item.id: item for item in get_texts_state(db).texts}
    assert restored.skipped is False
    assert texts[first.id].status == "queued"
    assert texts[first.id].skipped_at is None
    assert texts[first.id].queue_position == 1
    assert texts[second.id].queue_position == 2
    assert get_reader_state(db).lesson_id == first.id
    assert len(unread_lessons(db)) == 2


def test_started_lesson_returns_to_ready_until_new_activity_after_restore(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    record_events(
        db,
        [event_factory(lesson.id, "lesson.started", event_id="started-before-skip")],
    )

    record_lesson_queue_action(db, lesson_id=lesson.id, action_id=uuid4(), skipped=True)
    skipped = get_texts_state(db).texts[0]
    assert skipped.status == "skipped"
    assert skipped.session_count == 1
    assert skipped.opened_at is not None

    record_lesson_queue_action(db, lesson_id=lesson.id, action_id=uuid4(), skipped=False)
    restored = get_texts_state(db).texts[0]
    assert restored.status == "queued"
    assert restored.queue_position == 1
    assert restored.session_count == 1
    assert get_reader_state(db).lesson_id == lesson.id

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="reveal-after-restore",
                session_id="session-2",
                payload={"term_key": "es:manana:NOUN"},
            )
        ],
    )
    resumed = get_texts_state(db).texts[0]
    assert resumed.status == "in_progress"
    assert resumed.queue_position is None
    assert resumed.session_count == 2


def test_lesson_with_queue_history_cannot_be_replaced(db: Session, lesson_factory: Any) -> None:
    payload = lesson_factory()
    lesson = import_lesson(db, payload)
    record_lesson_queue_action(db, lesson_id=lesson.id, action_id=uuid4(), skipped=True)
    payload["title"] = "Replacement content"

    with pytest.raises(LessonConflictError, match="queue actions"):
        import_lesson(db, payload, replace=True)


def test_queue_actions_are_idempotent_conflict_checked_and_completed_lessons_cannot_skip(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    first = import_lesson(db, lesson_factory(key="first"))
    second = import_lesson(db, lesson_factory(key="second"))
    action_id = uuid4()

    original = record_lesson_queue_action(db, lesson_id=first.id, action_id=action_id, skipped=True)
    replay = record_lesson_queue_action(db, lesson_id=first.id, action_id=action_id, skipped=True)
    assert replay == original
    assert db.scalar(select(func.count()).select_from(LessonQueueAction)) == 1
    with pytest.raises(LessonQueueConflictError, match="action_id"):
        record_lesson_queue_action(db, lesson_id=first.id, action_id=action_id, skipped=False)
    with pytest.raises(LessonQueueConflictError, match="action_id"):
        record_lesson_queue_action(db, lesson_id=second.id, action_id=action_id, skipped=True)

    record_events(
        db,
        [
            event_factory(
                second.id,
                "lesson.completed",
                event_id="complete-before-skip",
                payload={"active_seconds": 60, "completion_ratio": 1},
            )
        ],
    )
    with pytest.raises(LessonQueueConflictError, match="completed"):
        record_lesson_queue_action(db, lesson_id=second.id, action_id=uuid4(), skipped=True)
    with pytest.raises(LessonQueueConflictError, match="restore a completed"):
        record_lesson_queue_action(db, lesson_id=second.id, action_id=uuid4(), skipped=False)

    # Defensive replay keeps older, already accepted restore rows from reviving completed content.
    db.add(
        LessonQueueAction(
            action_id=str(uuid4()),
            lesson_id=second.id,
            skipped=False,
        )
    )
    db.commit()
    completed = {item.id: item for item in get_texts_state(db).texts}[second.id]
    assert completed.status == "read"
    assert completed.queue_position is None


def test_skipped_targets_and_grammar_are_available_to_generation_again(
    db: Session, lesson_factory: Any
) -> None:
    update_profile(db, {"level": "A1"})
    lesson = import_lesson(db, _with_grammar(lesson_factory(key="opportunities")))
    before = build_agent_brief(db)
    before_term = next(term for term in before.priority_terms if term.key == "es:manana:NOUN")
    assert all(item.key != "es:present-indicative" for item in before.priority_grammar)

    record_lesson_queue_action(db, lesson_id=lesson.id, action_id=uuid4(), skipped=True)
    after = build_agent_brief(db)
    after_term = next(term for term in after.priority_terms if term.key == "es:manana:NOUN")

    assert after_term.urgency < before_term.urgency
    assert any(item.key == "es:present-indicative" for item in after.priority_grammar)
    assert after.recent_lessons[0].id == lesson.id


def test_skip_and_restore_api_scope_refill_and_conflicts(
    api_client: TestClient,
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "1")
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "2")
    update_profile(db, {"level": "A1"})
    first = import_lesson(db, lesson_factory(key="first"))
    second = import_lesson(db, lesson_factory(key="second"))
    action_id = str(uuid4())

    skipped = api_client.post(
        f"/api/profiles/es-es/texts/{first.id}/skip",
        json={"action_id": action_id},
    )
    replay = api_client.post(
        f"/api/profiles/es-es/texts/{first.id}/skip",
        json={"action_id": action_id},
    )
    conflict = api_client.post(
        f"/api/profiles/es-es/texts/{first.id}/restore",
        json={"action_id": action_id},
    )
    missing = api_client.post(
        "/api/profiles/es-es/texts/999/skip",
        json={"action_id": str(uuid4())},
    )

    assert skipped.status_code == replay.status_code == 200
    assert skipped.json() == replay.json()
    assert skipped.json()["lesson_id"] == first.id
    assert skipped.json()["skipped"] is True
    assert conflict.status_code == 409
    assert missing.status_code == 404
    assert db.scalar(select(func.count()).select_from(LessonQueueAction)) == 1
    task = db.scalar(select(GenerationTask))
    assert task is not None
    assert task.payload["unread_lesson_count"] == 1
    assert task.payload["queue_shortfall"] == 1
    assert task.payload["trigger_queue_action_ids"] == [action_id]
    assert db.scalar(select(func.count()).select_from(Interaction)) == 0

    restored = api_client.post(
        f"/api/profiles/es-es/texts/{first.id}/restore",
        json={"action_id": str(uuid4())},
    )
    assert restored.status_code == 200
    assert restored.json()["skipped"] is False

    record_events(
        db,
        [
            event_factory(
                second.id,
                "lesson.completed",
                event_id="api-completed",
                payload={"active_seconds": 60, "completion_ratio": 1},
            )
        ],
    )
    completed = api_client.post(
        f"/api/profiles/es-es/texts/{second.id}/skip",
        json={"action_id": str(uuid4())},
    )
    assert completed.status_code == 409


def test_fresh_skip_recovers_terminal_generation_but_replay_does_not_loop(
    api_client: TestClient,
    db: Session,
    lesson_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "1")
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "3")
    update_profile(db, {"level": "A1"})
    first_lesson = import_lesson(db, lesson_factory(key="first-terminal-skip"))
    import_lesson(db, lesson_factory(key="second-terminal-skip"))
    original = ensure_generation_task(db, require_enabled=False)
    assert original is not None

    first_attempt = claim_generation_task(db)
    assert first_attempt is not None
    fail_generation_task(
        db,
        first_attempt.id,
        log_path=str(tmp_path / "first.log"),
        error="first failure",
    )
    maintain_generation_task(db)
    second_attempt = claim_generation_task(db)
    assert second_attempt is not None
    fail_generation_task(
        db,
        second_attempt.id,
        log_path=str(tmp_path / "second.log"),
        error="second failure",
    )
    stopped = maintain_generation_task(db)
    assert stopped is not None and stopped.id == original.id

    action_id = str(uuid4())
    skipped = api_client.post(
        f"/api/profiles/es-es/texts/{first_lesson.id}/skip",
        json={"action_id": action_id},
    )

    assert skipped.status_code == 200
    replacement = db.scalar(select(GenerationTask).order_by(GenerationTask.id.desc()))
    assert replacement is not None and replacement.id != original.id
    assert replacement.payload["trigger_queue_action_ids"] == [action_id]

    replacement_first = claim_generation_task(db)
    assert replacement_first is not None
    fail_generation_task(
        db,
        replacement_first.id,
        log_path=str(tmp_path / "replacement-first.log"),
        error="replacement first failure",
    )
    maintain_generation_task(db)
    replacement_second = claim_generation_task(db)
    assert replacement_second is not None
    fail_generation_task(
        db,
        replacement_second.id,
        log_path=str(tmp_path / "replacement-second.log"),
        error="replacement second failure",
    )

    replay = api_client.post(
        f"/api/profiles/es-es/texts/{first_lesson.id}/skip",
        json={"action_id": action_id},
    )

    assert replay.status_code == 200
    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 2
    latest = db.scalar(select(GenerationTask).order_by(GenerationTask.id.desc()))
    assert latest is not None and latest.id == replacement.id and latest.state == "failed"


def test_maintenance_recovers_a_committed_skip_when_enqueue_wakeup_was_disabled(
    api_client: TestClient,
    db: Session,
    lesson_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "0")
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "3")
    update_profile(db, {"level": "A1"})
    lesson = import_lesson(db, lesson_factory(key="missed-skip-wakeup"))
    original = ensure_generation_task(db, require_enabled=False)
    assert original is not None
    first = claim_generation_task(db)
    assert first is not None
    fail_generation_task(db, first.id, log_path=str(tmp_path / "first.log"), error="first")
    maintain_generation_task(db)
    second = claim_generation_task(db)
    assert second is not None
    fail_generation_task(db, second.id, log_path=str(tmp_path / "second.log"), error="second")
    stopped = maintain_generation_task(db)
    assert stopped is not None and stopped.id == original.id

    action_id = str(uuid4())
    skipped = api_client.post(
        f"/api/profiles/es-es/texts/{lesson.id}/skip",
        json={"action_id": action_id},
    )
    assert skipped.status_code == 200
    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 1

    replacement = maintain_generation_task(db)

    assert replacement is not None and replacement.id != original.id
    assert replacement.payload["trigger_queue_action_cursor"] > 0
    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 2


def test_queue_action_rejects_a_lesson_outside_the_profile_languages(
    db: Session, lesson_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    update_profile(db, {"learning_language": "fr"})

    with pytest.raises(LookupError, match="lesson not found"):
        record_lesson_queue_action(db, lesson_id=lesson.id, action_id=uuid4(), skipped=True)


def test_ready_lessons_can_be_moved_and_reader_uses_the_durable_order(
    db: Session, lesson_factory: Any
) -> None:
    first = import_lesson(db, lesson_factory(key="move-first"))
    second = import_lesson(db, lesson_factory(key="move-second"))
    third = import_lesson(db, lesson_factory(key="move-third"))

    first_move_id = uuid4()
    first_move = record_lesson_queue_move(
        db,
        lesson_id=third.id,
        action_id=first_move_id,
        direction="up",
        neighbor_lesson_id=second.id,
    )
    record_lesson_queue_move(
        db,
        lesson_id=first.id,
        action_id=uuid4(),
        direction="down",
        neighbor_lesson_id=third.id,
    )

    texts = {item.id: item for item in get_texts_state(db).texts}
    assert texts[third.id].queue_position == 1
    assert texts[first.id].queue_position == 2
    assert texts[second.id].queue_position == 3
    assert get_reader_state(db).lesson_id == third.id

    replay = record_lesson_queue_move(
        db,
        lesson_id=third.id,
        action_id=first_move_id,
        direction="up",
        neighbor_lesson_id=second.id,
    )
    assert replay == first_move
    assert db.scalar(select(func.count()).select_from(LessonQueueMoveAction)) == 2
    assert db.scalar(select(func.count()).select_from(Interaction)) == 0


def test_queue_move_rejects_stale_edges_non_ready_lessons_and_action_id_conflicts(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    first = import_lesson(db, lesson_factory(key="move-guard-first"))
    second = import_lesson(db, lesson_factory(key="move-guard-second"))
    third = import_lesson(db, lesson_factory(key="move-guard-third"))

    with pytest.raises(LessonQueueConflictError, match="cannot move up"):
        record_lesson_queue_move(
            db,
            lesson_id=first.id,
            action_id=uuid4(),
            direction="up",
            neighbor_lesson_id=second.id,
        )
    with pytest.raises(LessonQueueConflictError, match="no longer adjacent"):
        record_lesson_queue_move(
            db,
            lesson_id=third.id,
            action_id=uuid4(),
            direction="up",
            neighbor_lesson_id=first.id,
        )

    action_id = uuid4()
    record_lesson_queue_move(
        db,
        lesson_id=second.id,
        action_id=action_id,
        direction="down",
        neighbor_lesson_id=third.id,
    )
    with pytest.raises(LessonQueueConflictError, match="action_id"):
        record_lesson_queue_move(
            db,
            lesson_id=second.id,
            action_id=action_id,
            direction="up",
            neighbor_lesson_id=first.id,
        )
    with pytest.raises(LessonQueueConflictError, match="action_id"):
        record_lesson_queue_move(
            db,
            lesson_id=first.id,
            action_id=action_id,
            direction="down",
            neighbor_lesson_id=third.id,
        )

    record_events(
        db,
        [event_factory(first.id, "lesson.started", event_id="open-before-move")],
    )
    with pytest.raises(LessonQueueConflictError, match="only Ready"):
        record_lesson_queue_move(
            db,
            lesson_id=first.id,
            action_id=uuid4(),
            direction="down",
            neighbor_lesson_id=third.id,
        )


def test_queue_moves_replay_before_skip_completion_and_later_import_filtering(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    first = import_lesson(db, lesson_factory(key="latent-first"))
    second = import_lesson(db, lesson_factory(key="latent-second"))
    third = import_lesson(db, lesson_factory(key="latent-third"))
    fourth = import_lesson(db, lesson_factory(key="latent-fourth"))
    record_lesson_queue_move(
        db,
        lesson_id=third.id,
        action_id=uuid4(),
        direction="up",
        neighbor_lesson_id=second.id,
    )

    record_lesson_queue_action(db, lesson_id=third.id, action_id=uuid4(), skipped=True)
    fifth = import_lesson(db, lesson_factory(key="latent-fifth"))
    skipped = {item.id: item for item in get_texts_state(db).texts}
    assert skipped[third.id].queue_position is None
    assert [
        lesson_id for lesson_id, item in skipped.items() if item.queue_position is not None
    ] == [first.id, second.id, fourth.id, fifth.id]
    assert [
        skipped[item].queue_position for item in (first.id, second.id, fourth.id, fifth.id)
    ] == [
        1,
        2,
        3,
        4,
    ]

    record_lesson_queue_action(db, lesson_id=third.id, action_id=uuid4(), skipped=False)
    restored = {item.id: item for item in get_texts_state(db).texts}
    assert [restored[item].queue_position for item in (first.id, third.id, second.id)] == [1, 2, 3]

    record_events(
        db,
        [
            event_factory(
                first.id,
                "lesson.completed",
                event_id="complete-latent-first",
                payload={"active_seconds": 60, "completion_ratio": 1},
            )
        ],
    )
    assert get_reader_state(db).lesson_id == third.id


def test_lesson_with_queue_move_history_cannot_be_replaced(
    db: Session, lesson_factory: Any
) -> None:
    first_payload = lesson_factory(key="move-replace-first")
    second_payload = lesson_factory(key="move-replace-second")
    first = import_lesson(db, first_payload)
    second = import_lesson(db, second_payload)
    record_lesson_queue_move(
        db,
        lesson_id=second.id,
        action_id=uuid4(),
        direction="up",
        neighbor_lesson_id=first.id,
    )

    first_payload["title"] = "Replacement first"
    second_payload["title"] = "Replacement second"
    with pytest.raises(LessonConflictError, match="queue moves"):
        import_lesson(db, first_payload, replace=True)
    with pytest.raises(LessonConflictError, match="queue moves"):
        import_lesson(db, second_payload, replace=True)


def test_queue_move_api_is_idempotent_and_reports_stale_or_missing_lessons(
    api_client: TestClient, db: Session, lesson_factory: Any
) -> None:
    first = import_lesson(db, lesson_factory(key="api-move-first"))
    second = import_lesson(db, lesson_factory(key="api-move-second"))
    third = import_lesson(db, lesson_factory(key="api-move-third"))
    action_id = str(uuid4())
    body = {
        "action_id": action_id,
        "direction": "up",
        "neighbor_lesson_id": first.id,
    }

    moved = api_client.post(
        f"/api/profiles/es-es/texts/{second.id}/move",
        json=body,
    )
    replay = api_client.post(
        f"/api/profiles/es-es/texts/{second.id}/move",
        json=body,
    )
    stale = api_client.post(
        f"/api/profiles/es-es/texts/{third.id}/move",
        json={
            "action_id": str(uuid4()),
            "direction": "up",
            "neighbor_lesson_id": second.id,
        },
    )
    missing = api_client.post(
        "/api/profiles/es-es/texts/999/move",
        json={
            "action_id": str(uuid4()),
            "direction": "up",
            "neighbor_lesson_id": second.id,
        },
    )

    assert moved.status_code == replay.status_code == 200
    assert moved.json() == replay.json()
    assert moved.json()["lesson_id"] == second.id
    assert moved.json()["neighbor_lesson_id"] == first.id
    assert moved.json()["direction"] == "up"
    assert stale.status_code == 409
    assert missing.status_code == 404
    state = api_client.get("/api/profiles/es-es/texts").json()
    positions = {item["id"]: item["queue_position"] for item in state["texts"]}
    assert positions == {second.id: 1, first.id: 2, third.id: 3}


def test_concurrent_queue_moves_serialize_validation_and_append(
    session_factory: sessionmaker[Session], lesson_factory: Any, tmp_path: Path
) -> None:
    workspace = Workspace(
        profile_id="es-es",
        label="Spanish",
        learning_language="es-ES",
        translation_language="en",
        directory=tmp_path,
        database_path=tmp_path / "test.db",
        relative_directory=".",
        relative_database="test.db",
    )
    with session_factory() as setup:
        setup.info["workspace"] = workspace
        ensure_profile(setup)
        first = import_lesson(setup, lesson_factory(key="concurrent-move-first"))
        second = import_lesson(setup, lesson_factory(key="concurrent-move-second"))
        third = import_lesson(setup, lesson_factory(key="concurrent-move-third"))

    start = Barrier(2)

    def move(lesson_id: int, neighbor_lesson_id: int) -> str:
        with session_factory() as db:
            db.info["workspace"] = workspace
            start.wait()
            try:
                record_lesson_queue_move(
                    db,
                    lesson_id=lesson_id,
                    action_id=uuid4(),
                    direction="up",
                    neighbor_lesson_id=neighbor_lesson_id,
                )
            except LessonQueueConflictError:
                return "conflict"
            return "moved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = [
            executor.submit(move, second.id, first.id),
            executor.submit(move, third.id, second.id),
        ]
    assert sorted(result.result() for result in outcomes) == ["conflict", "moved"]

    with session_factory() as check:
        check.info["workspace"] = workspace
        actions = check.scalar(select(func.count()).select_from(LessonQueueMoveAction))
        positions = {
            item.id: item.queue_position
            for item in get_texts_state(check).texts
            if item.queue_position is not None
        }
    assert actions == 1
    assert positions in (
        {second.id: 1, first.id: 2, third.id: 3},
        {first.id: 1, third.id: 2, second.id: 3},
    )
    assert (workspace.directory / "queue-moves.lock").stat().st_mode & 0o777 == 0o600
