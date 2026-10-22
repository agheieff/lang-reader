from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.agent_brief import calibration_generation_brief
from server.db import init_db, session_scope
from server.generation_queue import (
    claim_generation_task,
    ensure_generation_task,
    request_topic_lesson,
)
from server.learning import import_lesson
from server.models import AudioTask, GenerationTask, Lesson
from server.profile_activation import (
    ACTIVATION_PREFERENCE_KEY,
    ProfileActivationUpdate,
    activate_profile_settings,
    profile_is_active,
)
from server.profile_state import (
    ProfileInactiveError,
    ensure_profile,
    profile_level_source,
    update_profile,
)
from server.schemas import TextRequestIn
from server.workspaces import WorkspaceRegistry


def _deactivate(db: Session) -> None:
    profile = ensure_profile(db)
    preferences = dict(profile.preferences)
    preferences[ACTIVATION_PREFERENCE_KEY] = {
        "schema_version": 1,
        "active": False,
        "activated_at": None,
    }
    profile.preferences = preferences
    db.commit()


def test_inactive_profile_cannot_generate_or_request_content(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _deactivate(db)
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "1")

    assert ensure_generation_task(db) is None
    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 0
    with pytest.raises(ProfileInactiveError, match="activate"):
        request_topic_lesson(
            db,
            TextRequestIn(request_id="d3c65db7-6eaf-48e7-820b-e25ca1c8c503", topic="Trains"),
        )

    dormant = GenerationTask(
        state="pending",
        dedupe_key="lesson-queue",
        payload={"schema_version": 1},
    )
    db.add(dormant)
    db.commit()
    assert claim_generation_task(db) is None
    assert dormant.state == "pending"


@pytest.mark.parametrize(
    ("starting_point", "profile_seed", "upper_bound"),
    [
        ("complete_beginner", 0.02, 0.45),
        ("near_native", 0.94, 0.98),
    ],
)
def test_questionnaire_seeds_broad_first_calibration_without_replacing_it(
    db: Session,
    starting_point: str,
    profile_seed: float,
    upper_bound: float,
) -> None:
    _deactivate(db)
    view = activate_profile_settings(
        db,
        ProfileActivationUpdate.model_validate(
            {
                "starting_point": starting_point,
                "confidence": "high",
                "interests": ["science fiction", "history"],
                "text_length": 240,
            }
        ),
    )
    profile = ensure_profile(db)
    task = ensure_generation_task(db, require_enabled=False)
    brief = calibration_generation_brief(db)

    assert view.active
    assert view.questionnaire_completed
    assert profile_is_active(profile)
    assert profile_level_source(profile) == "unknown"
    assert profile.difficulty == pytest.approx(profile_seed)
    assert profile.interests == ["science fiction", "history"]
    assert profile.preferences["text_length"] == 240
    assert task is not None
    assert task.payload["generation_mode"] == "calibration"
    assert brief is not None
    assert brief.sequence == 1
    # The declared answer remains the one current center. The truncated Gaussian contributes only
    # the uncertainty used to spread the probes until reading evidence arrives.
    assert brief.target_difficulty == pytest.approx(profile_seed)
    assert min(brief.probe_difficulties) >= 0.02
    assert max(brief.probe_difficulties) <= upper_bound


def test_new_workspace_initialization_is_inactive_and_has_no_background_work(
    tmp_path: Path,
) -> None:
    registry = WorkspaceRegistry(tmp_path)
    workspace = registry.create(
        "fresh-de",
        learning_language="de-DE",
        translation_language="en",
    )

    init_db(workspace)

    with session_scope(workspace) as db:
        profile = ensure_profile(db)
        assert not profile_is_active(profile)
        assert db.scalar(select(func.count()).select_from(Lesson)) == 0
        assert db.scalar(select(func.count()).select_from(GenerationTask)) == 0
        assert db.scalar(select(func.count()).select_from(AudioTask)) == 0


def test_activation_is_idempotent_and_settings_cannot_forge_it(db: Session) -> None:
    _deactivate(db)
    first = activate_profile_settings(
        db,
        ProfileActivationUpdate(starting_point="simple_texts", confidence="medium"),
    )
    second = activate_profile_settings(
        db,
        ProfileActivationUpdate(starting_point="near_native", confidence="high"),
    )
    profile = ensure_profile(db)
    original_activation = profile.preferences[ACTIVATION_PREFERENCE_KEY]

    update_profile(
        db,
        {
            "preferences": {
                ACTIVATION_PREFERENCE_KEY: {"active": False},
                "text_length": 500,
            }
        },
    )
    profile = ensure_profile(db)

    assert first == second
    assert profile.difficulty == pytest.approx(0.25)
    assert profile.preferences[ACTIVATION_PREFERENCE_KEY] == original_activation
    assert profile.preferences["text_length"] == 500


def test_questionnaire_replaces_a_dormant_placeholder_as_a_revisable_prior(
    db: Session,
) -> None:
    update_profile(db, {"level": "C1", "difficulty": 0.75})
    _deactivate(db)

    activate_profile_settings(
        db,
        ProfileActivationUpdate(starting_point="simple_texts", confidence="medium"),
    )
    profile = ensure_profile(db)

    assert profile.difficulty == pytest.approx(0.25)
    assert profile_level_source(profile) == "unknown"


def test_activation_without_questionnaire_preserves_a_self_reported_level(db: Session) -> None:
    update_profile(db, {"level": "C1", "difficulty": 0.75})
    _deactivate(db)

    activate_profile_settings(db, ProfileActivationUpdate())
    profile = ensure_profile(db)

    assert profile.difficulty == pytest.approx(0.75)
    assert profile_level_source(profile) == "self_reported"


def test_activation_api_is_explicit_and_starts_no_audio_without_lessons(
    api_client: TestClient,
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deactivate(db)
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "0")

    before = api_client.get("/api/profiles/es-es/activation")
    response = api_client.post(
        "/api/profiles/es-es/activation",
        json={"starting_point": "unsure", "interests": []},
    )

    assert before.status_code == 200
    assert before.json()["active"] is False
    assert response.status_code == 200
    assert response.json()["active"] is True
    assert response.json()["questionnaire_completed"] is False
    assert db.scalar(select(func.count()).select_from(Lesson)) == 0
    assert db.scalar(select(func.count()).select_from(AudioTask)) == 0


def test_existing_profile_activation_migration_uses_real_content(
    db: Session, lesson_factory: Any
) -> None:
    profile = ensure_profile(db)
    preferences = dict(profile.preferences)
    preferences.pop(ACTIVATION_PREFERENCE_KEY)
    profile.preferences = preferences
    db.commit()
    ensure_profile(db)
    assert not profile_is_active(profile)

    preferences = dict(profile.preferences)
    preferences[ACTIVATION_PREFERENCE_KEY] = {
        "schema_version": 1,
        "active": True,
        "activated_at": None,
    }
    profile.preferences = preferences
    db.commit()
    import_lesson(db, lesson_factory())
    preferences = dict(profile.preferences)
    preferences.pop(ACTIVATION_PREFERENCE_KEY)
    profile.preferences = preferences
    db.commit()

    migrated = ensure_profile(db)
    assert profile_is_active(migrated)
