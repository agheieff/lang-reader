from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.generation_queue import (
    claim_generation_task,
    defer_generation_task,
    ensure_generation_task,
    request_topic_lesson,
)
from server.learning import get_reader_state, import_lesson, record_events
from server.models import GenerationTask, Interaction, LexemeState
from server.schemas import TextRequestIn
from server.texts import get_text_detail, get_texts_state
from server.words import get_words_state


def test_text_library_groups_queue_progress_and_read_activity(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    read = import_lesson(db, lesson_factory(key="read"))
    started = import_lesson(db, lesson_factory(key="started"))
    queued = import_lesson(db, lesson_factory(key="queued"))
    record_events(
        db,
        [
            event_factory(
                read.id,
                "lesson.completed",
                event_id="read-complete-one",
                session_id="read-one",
                payload={"active_seconds": 45, "completion_ratio": 1},
            ),
            event_factory(
                read.id,
                "lesson.rated",
                event_id="read-rating",
                session_id="read-one",
                payload={"rating": 1},
                seconds=1,
            ),
            event_factory(
                read.id,
                "lesson.completed",
                event_id="read-complete-two",
                session_id="read-two",
                payload={"active_seconds": 45, "completion_ratio": 1},
                seconds=2,
            ),
            event_factory(
                started.id,
                "lesson.started",
                event_id="started-open",
                session_id="started-one",
                seconds=3,
            ),
        ],
    )

    state = get_texts_state(db)
    by_key = {text.key: text for text in state.texts}

    assert [text.id for text in state.texts] == [read.id, started.id, queued.id]
    assert by_key["read"].status == "read"
    assert by_key["read"].queue_position is None
    assert by_key["read"].session_count == 2
    assert by_key["read"].completion_count == 2
    assert by_key["read"].rating == 1
    assert by_key["started"].status == "in_progress"
    assert by_key["started"].queue_position is None
    assert by_key["queued"].status == "queued"
    assert by_key["queued"].queue_position == 1
    assert by_key["queued"].opened_at is None
    assert all(text.lexical_token_count == 3 for text in state.texts)


def test_text_library_projects_active_generation_without_inventing_lesson_details(
    db: Session, lesson_factory: Any
) -> None:
    queue_task = ensure_generation_task(db, require_enabled=False)
    assert queue_task is not None
    topic_task = request_topic_lesson(
        db,
        TextRequestIn(request_id=uuid4(), topic="an overnight train"),
    )

    pending = get_texts_state(db)
    by_task = {preparation.task_id: preparation for preparation in pending.preparations}
    assert set(by_task) == {queue_task.id, topic_task.id}
    assert by_task[queue_task.id].state == "pending"
    assert by_task[queue_task.id].request_kind == "queue_fill"
    assert by_task[queue_task.id].requested_topic is None
    assert by_task[topic_task.id].request_kind == "topic_request"
    assert by_task[topic_task.id].requested_topic == "an overnight train"

    claimed = claim_generation_task(db)
    assert claimed is not None
    assert claimed.id == queue_task.id
    running = get_texts_state(db)
    assert running.preparations[0].task_id == queue_task.id
    assert running.preparations[0].state == "running"

    import_lesson(db, lesson_factory(key=f"generated-task-{queue_task.id}-1"))
    imported = get_texts_state(db)
    assert [preparation.task_id for preparation in imported.preparations] == [topic_task.id]


def test_text_preparation_says_when_it_waits_for_model_capacity(db: Session) -> None:
    task = ensure_generation_task(db, require_enabled=False)
    assert task is not None
    assert get_texts_state(db).preparations[0].waiting_for_capacity is False

    claimed = claim_generation_task(db)
    assert claimed is not None
    defer_generation_task(db, claimed.id, log_path="x.log", reason="capacity unavailable")
    assert get_texts_state(db).preparations[0].waiting_for_capacity is True

    # A later transition (here a plain failure retry) is no longer a deferral.
    stored = db.get(GenerationTask, task.id)
    assert stored is not None
    stored.updated_at = datetime.fromisoformat(stored.payload["admission_retry_at"]) + timedelta(
        seconds=1
    )
    db.commit()
    assert get_texts_state(db).preparations[0].waiting_for_capacity is False


def test_text_preview_is_session_free_and_creates_no_learning_evidence(
    db: Session, lesson_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    before_states = [
        (state.term_key, state.alpha, state.beta, state.qualified_exposures)
        for state in db.scalars(select(LexemeState).order_by(LexemeState.term_key)).all()
    ]

    detail = get_text_detail(db, lesson.id)

    after_states = [
        (state.term_key, state.alpha, state.beta, state.qualified_exposures)
        for state in db.scalars(select(LexemeState).order_by(LexemeState.term_key)).all()
    ]
    assert detail.lesson_id == lesson.id
    assert detail.lesson.key == "lesson-one"
    assert detail.term_bands
    assert db.scalar(select(func.count()).select_from(Interaction)) == 0
    assert before_states == after_states
    assert get_words_state(db).words == []


def test_fresh_reader_returns_blank_progress_without_changing_default_resume(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="old-reveal",
                session_id="old-session",
                payload={"term_key": "es:manana:NOUN"},
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="old-completion",
                session_id="old-session",
                payload={"active_seconds": 60, "completion_ratio": 1},
                seconds=60,
            ),
        ],
    )

    resumed = get_reader_state(db, lesson_id=lesson.id)
    fresh = get_reader_state(db, lesson_id=lesson.id, fresh=True)

    assert resumed.progress.session_id == "old-session"
    assert resumed.progress.completed is True
    assert resumed.progress.revealed_term_keys == ["es:manana:NOUN"]
    assert fresh.lesson_id == lesson.id
    assert fresh.progress.session_id is None
    assert fresh.progress.started is False
    assert fresh.progress.completed is False
    assert fresh.progress.revealed_term_keys == []
    assert get_reader_state(db).lesson is None


def test_text_api_preview_fresh_reader_and_idempotent_topic_request(
    api_client: TestClient, db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="api-read-complete",
                session_id="api-old-session",
                payload={"active_seconds": 60, "completion_ratio": 1},
            )
        ],
    )
    interaction_count = db.scalar(select(func.count()).select_from(Interaction))

    listing = api_client.get("/api/profiles/es-es/texts")
    preview = api_client.get(f"/api/profiles/es-es/texts/{lesson.id}")
    missing = api_client.get("/api/profiles/es-es/texts/999")
    fresh = api_client.get(
        "/api/profiles/es-es/reader", params={"lesson_id": lesson.id, "fresh": "true"}
    )
    request_id = str(uuid4())
    first = api_client.post(
        "/api/profiles/es-es/text-requests",
        json={"request_id": request_id, "topic": "  deep-sea exploration  "},
    )
    replay = api_client.post(
        "/api/profiles/es-es/text-requests",
        json={"request_id": request_id, "topic": "deep-sea exploration"},
    )
    conflict = api_client.post(
        "/api/profiles/es-es/text-requests",
        json={"request_id": request_id, "topic": "urban architecture"},
    )

    assert listing.status_code == 200
    assert listing.json()["texts"][0]["status"] == "read"
    assert preview.status_code == 200
    assert preview.json()["lesson_id"] == lesson.id
    assert "progress" not in preview.json()
    assert missing.status_code == 404
    assert fresh.status_code == 200
    assert fresh.json()["progress"]["session_id"] is None
    assert first.status_code == replay.status_code == 202
    assert first.json()["task_id"] == replay.json()["task_id"]
    assert first.json()["topic"] == "deep-sea exploration"
    assert conflict.status_code == 409
    assert db.scalar(select(func.count()).select_from(Interaction)) == interaction_count

    updated = api_client.get("/api/profiles/es-es/texts").json()
    assert updated["requests"][0]["request_id"] == request_id
    assert updated["requests"][0]["state"] == "pending"
    assert updated["preparations"][0]["request_kind"] == "topic_request"
    assert updated["preparations"][0]["requested_topic"] == "deep-sea exploration"
