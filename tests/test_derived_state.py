from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from server.db import SCHEMA_VERSION, migrate_database
from server.derived_state import ensure_derived_states
from server.generation_queue import maintain_generation_task
from server.learning import import_lesson, record_events
from server.models import Base, DerivedState, LexemeState, ProficiencyState, Profile
from server.profile_activation import activate_profile_settings
from server.profile_state import update_profile


def test_fresh_database_is_created_at_the_current_version(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    assert migrate_database(engine) == SCHEMA_VERSION
    with sqlite3.connect(tmp_path / "fresh.db") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        columns = {row[1] for row in connection.execute("PRAGMA table_info(proficiency_state)")}
    assert "qualified_readings" in columns


def test_unversioned_database_is_migrated_and_keeps_its_evidence(tmp_path: Path) -> None:
    path = tmp_path / "baseline.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Profile(id=1, learning_language="zh-Hans"))
        session.commit()
    with sqlite3.connect(path) as connection:  # the pre-migration proficiency cache shape
        connection.execute("DROP TABLE proficiency_state")
        connection.execute("CREATE TABLE proficiency_state (id INTEGER PRIMARY KEY, status TEXT)")
        connection.execute("PRAGMA user_version = 0")

    migrate_database(engine)
    migrate_database(engine)  # idempotent for the second process that starts

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT learning_language FROM profile").fetchall() == [
            ("zh-Hans",)
        ]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(proficiency_state)")}
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert "qualified_readings" in columns


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "newer.db"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="newer"):
        migrate_database(create_engine(f"sqlite:///{path}"))


def test_derived_states_rebuild_only_after_new_evidence(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    assert ensure_derived_states(db) is False  # import already replayed
    rebuilt_at = db.get(DerivedState, 1).rebuilt_at  # type: ignore[union-attr]

    record_events(
        db,
        [event_factory(lesson.id, "lesson.started", event_id="started")],
        rebuild_derived=False,
    )
    assert ensure_derived_states(db) is True
    assert ensure_derived_states(db) is False
    assert db.get(DerivedState, 1).rebuilt_at >= rebuilt_at  # type: ignore[union-attr]


def test_idle_maintenance_does_not_rewrite_proficiency(
    db: Session, lesson_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "1")
    activate_profile_settings(db, {"starting_point": "simple_texts"})
    update_profile(db, {"level": "B1", "difficulty": 0.5})
    import_lesson(db, lesson_factory())
    ensure_derived_states(db)
    before = db.scalar(select(ProficiencyState.updated_at))

    assert maintain_generation_task(db) is None  # one unread text satisfies the queue

    assert db.scalar(select(ProficiencyState.updated_at)) == before


def test_http_events_replay_after_the_response(
    api_client: TestClient, db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    term_key = next(iter(db.scalars(select(LexemeState.term_key))))
    event = event_factory(
        lesson.id, "term.revealed", event_id="reveal", payload={"term_key": term_key}
    )
    response = api_client.post("/api/profiles/es-es/events", json={"events": [event]})
    assert response.status_code == 200
    db.expire_all()
    state = db.get(LexemeState, ("es-ES", "en", term_key))
    assert state is not None and state.reveal_failures == 1
