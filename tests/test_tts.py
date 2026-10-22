from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qwen_provider import _validate_request
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.learning import import_lesson
from server.models import AudioTask, Lesson, Profile
from server.profile_activation import (
    ACTIVATION_PREFERENCE_KEY,
    ProfileActivationUpdate,
    activate_profile_settings,
)
from server.profile_fingerprint import profile_fingerprint
from server.profile_state import ensure_profile
from server.schemas import LessonDocument
from server.tts import (
    TTS_MODEL_ID,
    TTS_MODEL_REVISION,
    _failed_audio_reason,
    audio_cache_key,
    audio_status,
    ensure_audio_backfill,
    get_tts_settings,
    lesson_audio_chunks,
    model_language,
    recommended_voice_id,
    set_tts_voice,
)
from server.workspaces import Workspace


def test_lesson_import_enqueues_audio_and_voice_change_backfills_without_staling_generation(
    db: Session, lesson_factory: Any
) -> None:
    first = import_lesson(db, lesson_factory(key="audio-one"))
    second = import_lesson(db, lesson_factory(key="audio-two"))
    tasks = db.scalars(select(AudioTask).order_by(AudioTask.id)).all()

    assert [(task.lesson_id, task.voice_id, task.state) for task in tasks] == [
        (first.id, "Aiden", "pending"),
        (second.id, "Aiden", "pending"),
    ]
    fingerprint = profile_fingerprint(db)

    profile = set_tts_voice(db, "Serena")
    tasks = db.scalars(select(AudioTask).order_by(AudioTask.id)).all()

    assert get_tts_settings(profile).selected_voice_id == "Serena"
    assert profile_fingerprint(db) == fingerprint
    assert [task.state for task in tasks[:2]] == ["superseded", "superseded"]
    assert [(task.lesson_id, task.voice_id, task.state) for task in tasks[2:]] == [
        (first.id, "Serena", "pending"),
        (second.id, "Serena", "pending"),
    ]


def test_audio_cache_and_chunks_use_only_canonical_source_text(
    db: Session, lesson_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory(key="audio-source"))
    document = LessonDocument.model_validate(lesson.payload)
    chunks = lesson_audio_chunks(document)

    assert chunks[0] == "Lesson audio-source"
    assert any("mañana" in chunk for chunk in chunks)
    assert all("Translation" not in chunk for chunk in chunks)
    assert all("science" not in chunk for chunk in chunks)
    assert audio_cache_key(lesson, "Aiden") != audio_cache_key(lesson, "Serena")


def test_tts_api_exposes_profile_voice_and_never_accepts_arbitrary_text(
    api_client: TestClient, db: Session, lesson_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory(key="audio-api"))

    settings = api_client.get("/api/profiles/es-es/tts")
    updated = api_client.put("/api/profiles/es-es/tts", json={"voice_id": "Serena"})
    invalid = api_client.put("/api/profiles/es-es/tts", json={"voice_id": "unknown"})
    status = api_client.get(f"/api/profiles/es-es/lessons/{lesson.id}/audio/status")
    file_response = api_client.get(f"/api/profiles/es-es/lessons/{lesson.id}/audio")

    assert settings.status_code == 200
    assert settings.json()["selected_voice_id"] == "Aiden"
    assert updated.status_code == 200
    assert updated.json()["selected_voice_id"] == "Serena"
    assert invalid.status_code == 422
    assert status.status_code == 200
    assert status.json()["state"] in {"preparing", "unavailable"}
    expected_reason = {
        "preparing": "queued",
        "unavailable": "provider_unavailable",
    }[status.json()["state"]]
    assert status.json()["reason"] == expected_reason
    assert file_response.status_code == 404


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        ("TimeoutError: Qwen audio generation timed out", "generation_timeout"),
        ("ValueError: Qwen provider created an invalid WAV", "invalid_audio"),
        ("RuntimeError: Qwen provider stopped unexpectedly (1)", "provider_stopped"),
        ("RuntimeError: GPU failure in provider", "generation_failed"),
    ],
)
def test_audio_failures_expose_safe_reason_codes(error: str, reason: str) -> None:
    assert _failed_audio_reason(error) == reason


def test_audio_status_distinguishes_queue_generation_retry_and_failure(
    db: Session,
    lesson_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lesson = import_lesson(db, lesson_factory(key="audio-status-phases"))
    task = db.scalar(select(AudioTask).where(AudioTask.lesson_id == lesson.id))
    assert task is not None
    workspace = Workspace(
        profile_id="es-es",
        label="Spanish",
        learning_language="es-ES",
        translation_language="en",
        directory=tmp_path,
        database_path=tmp_path / "lang.db",
        relative_directory="profiles/es-es",
        relative_database="profiles/es-es/lang.db",
    )
    monkeypatch.setattr("server.tts.provider_command", lambda: ("provider",))

    assert audio_status(db, workspace, lesson.id).reason == "queued"
    task.state = "running"
    db.commit()
    assert audio_status(db, workspace, lesson.id).reason == "generating"

    task.state = "pending"
    task.attempts = 1
    task.error = "RuntimeError: first attempt"
    db.commit()
    assert audio_status(db, workspace, lesson.id).reason == "retrying"

    task.state = "failed"
    task.attempts = 2
    task.error = "TimeoutError: Qwen audio generation timed out"
    db.commit()
    status = audio_status(db, workspace, lesson.id)
    assert status.state == "failed"
    assert status.reason == "generation_timeout"
    assert "timed out" in (status.message or "")


def test_isolated_provider_rejects_model_paths_and_unbounded_text() -> None:
    valid = {
        "request_id": "audio-1",
        "model_id": TTS_MODEL_ID,
        "model_revision": TTS_MODEL_REVISION,
        "language": "Spanish",
        "voice_id": "Aiden",
        "chunks": ["Una frase breve."],
        "output_path": "/tmp/audio.tmp.wav",
    }
    _validate_request(valid)
    _validate_request({**valid, "language": "German", "voice_id": "Ryan"})

    for patch in (
        {"model_id": "/tmp/other-model"},
        {"language": "Cantonese"},
        {"chunks": ["x" * 501]},
        {"output_path": "/tmp/not-audio.txt"},
    ):
        with pytest.raises(ValueError):
            _validate_request({**valid, **patch})


def test_german_uses_qwen_german_mode_with_an_explicit_compatible_default() -> None:
    settings = get_tts_settings(Profile(learning_language="de-DE", preferences={}))

    assert model_language("de-DE") == "German"
    assert recommended_voice_id("de-DE") == "Ryan"
    assert settings.model_language == "German"
    assert settings.selected_voice_id == "Ryan"
    assert [voice.id for voice in settings.voices if voice.recommended] == ["Ryan"]
    assert settings.note is not None and "native-German" in settings.note


def test_existing_lesson_audio_task_is_idempotent(db: Session, lesson_factory: Any) -> None:
    lesson = import_lesson(db, lesson_factory(key="audio-idempotent"))
    duplicate = import_lesson(db, lesson_factory(key="audio-idempotent"))
    tasks = db.scalars(select(AudioTask).where(AudioTask.lesson_id == lesson.id)).all()

    assert duplicate.id == lesson.id
    assert len(tasks) == 1
    assert db.scalar(select(Lesson).where(Lesson.id == lesson.id)) is not None


def test_inactive_profile_import_and_status_never_enqueue_audio(
    db: Session,
    lesson_factory: Any,
    tmp_path: Path,
) -> None:
    profile = ensure_profile(db)
    preferences = dict(profile.preferences)
    preferences[ACTIVATION_PREFERENCE_KEY] = {
        "schema_version": 1,
        "active": False,
        "activated_at": None,
    }
    profile.preferences = preferences
    db.commit()

    lesson = import_lesson(db, lesson_factory(key="inactive-audio"))
    set_tts_voice(db, "Serena")
    workspace = Workspace(
        profile_id="es-es",
        label="Spanish",
        learning_language="es-ES",
        translation_language="en",
        directory=tmp_path,
        database_path=tmp_path / "lang.db",
        relative_directory="profiles/es-es",
        relative_database="profiles/es-es/lang.db",
    )
    status = audio_status(db, workspace, lesson.id)

    assert status.state == "disabled"
    assert status.reason == "profile_inactive"
    assert db.scalars(select(AudioTask)).all() == []

    activate_profile_settings(db, ProfileActivationUpdate())
    tasks = ensure_audio_backfill(db)
    assert [(task.lesson_id, task.voice_id, task.state) for task in tasks] == [
        (lesson.id, "Serena", "pending")
    ]
