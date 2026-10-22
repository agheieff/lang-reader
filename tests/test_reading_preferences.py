from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from server.agent_brief import build_agent_brief
from server.content_planning import MOVE_WEIGHTS, build_content_plan, choose_move
from server.db import init_db, session_scope
from server.generation_callback import CallbackInvocation, CallbackResult
from server.learning import import_lesson, record_events
from server.preference_agent import maybe_update_preferences
from server.profile_activation import ProfileActivationUpdate, activate_profile_settings
from server.profile_state import update_profile
from server.reading_preferences import (
    PreferencesConflictError,
    add_preference_message,
    content_history,
    current_preferences,
    pending_messages,
    preferences_view,
    save_user_preferences,
    update_due,
)
from server.schemas import PreferenceUpdateRequest
from server.workspaces import WorkspaceRegistry


def test_notes_start_from_interests_and_user_edits_are_revisioned(db: Session) -> None:
    update_profile(db, {"interests": ["history", "astronomy"]})
    assert "history, astronomy" in current_preferences(db)
    assert preferences_view(db).source == "default"

    saved = save_user_preferences(db, "Likes: history of science", expected_revision_id=None)
    assert saved.source == "user" and saved.revision_id is not None
    with pytest.raises(PreferencesConflictError):
        save_user_preferences(db, "stale", expected_revision_id=None)


def test_messages_are_idempotent_and_make_an_update_due(db: Session) -> None:
    assert not update_due(db)
    add_preference_message(db, "7a0c7c6e-6f0e-4d53-9d3b-8d3c1c2b1a00", "More sci-fi")
    add_preference_message(db, "7a0c7c6e-6f0e-4d53-9d3b-8d3c1c2b1a00", "More sci-fi")
    assert [message.text for message in pending_messages(db)] == ["More sci-fi"]
    assert update_due(db)
    with pytest.raises(PreferencesConflictError):
        add_preference_message(db, "7a0c7c6e-6f0e-4d53-9d3b-8d3c1c2b1a00", "Different")


def test_content_history_reports_moves_hypotheses_and_reactions(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    tested = import_lesson(
        db,
        lesson_factory(
            key="tested",
            metadata={
                "content_plan": {"move": "new"},
                "content_angle": "maritime history",
                "content_hypothesis": "Do you enjoy maritime history?",
            },
        ),
    )
    import_lesson(db, lesson_factory(key="untouched"))
    record_events(
        db,
        [
            event_factory(
                tested.id,
                "lesson.completed",
                event_id="done",
                payload={"active_seconds": 60, "completion_ratio": 1},
            ),
            event_factory(
                tested.id,
                "lesson.rated",
                event_id="rated",
                payload={"rating": -1, "feedback": ["new_topic"]},
                seconds=1,
            ),
        ],
    )

    history = content_history(db)
    assert [item.reaction for item in history] == ["unread", "disliked"]
    assert history[1].move == "new"
    assert history[1].hypothesis == "Do you enjoy maritime history?"
    assert history[1].feedback == ["new_topic"]
    brief = build_agent_brief(db)
    assert brief.content_history == history
    assert brief.reading_preferences


def test_moves_are_deterministic_and_follow_the_weights() -> None:
    assert choose_move(42) == choose_move(42)
    counts = Counter(choose_move(task_id) for task_id in range(3_000))
    for move, weight in MOVE_WEIGHTS:
        assert abs(counts[move] / 3_000 - weight) < 0.04


def test_plan_carries_the_move_and_requested_topics_override_it(db: Session) -> None:
    brief = build_agent_brief(db)
    plan = build_content_plan(brief, task_id=7, requested_topic=None)
    assert plan.move == choose_move(7)
    assert plan.move_instruction
    requested = build_content_plan(brief, task_id=7, requested_topic="lighthouses")
    assert requested.move == "requested" and "lighthouses" in requested.concrete_seed


def test_agent_update_folds_in_messages_and_respects_concurrent_edits(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        activate_profile_settings(db, ProfileActivationUpdate())
        add_preference_message(db, "0f5a1f30-86a1-4bb3-a3a2-6fb45a1d8c11", "Less cooking")
    seen: list[PreferenceUpdateRequest] = []

    class Agent:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            assert isinstance(invocation.request, PreferenceUpdateRequest)
            seen.append(invocation.request)
            payload = {"schema_version": 1, "preferences": "Dislikes: cooking", "reason": "msg"}
            return CallbackResult(exit_code=0, payload=json.dumps(payload), log="")

    assert maybe_update_preferences(workspace, Agent()) is True
    assert seen[0].user_messages == ["Less cooking"]
    with session_scope(workspace) as db:
        view = preferences_view(db)
        assert view.source == "agent" and "cooking" in view.text
        assert view.last_agent_reason == "msg" and not view.pending_messages
        assert not update_due(db)

    class EditingAgent(Agent):
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            with session_scope(workspace) as db:  # the learner saves while the agent runs
                save_user_preferences(
                    db, "Mine", expected_revision_id=preferences_view(db).revision_id
                )
            return super().run(invocation)

    with session_scope(workspace) as db:
        add_preference_message(db, "a86e0f7e-5c2a-4a1b-9b0e-2b4b8f4a6c22", "More poetry")
    assert maybe_update_preferences(workspace, EditingAgent()) is False
    with session_scope(workspace) as db:
        assert current_preferences(db).strip() == "Mine"
        assert len(pending_messages(db)) == 1  # retried later from the learner's revision


def test_preferences_api_round_trip(api_client: TestClient) -> None:
    base = "/api/profiles/es-es/reading-preferences"
    view = api_client.get(base).json()
    assert view["source"] == "default"

    saved = api_client.put(base, json={"text": "Likes: poetry", "expected_revision_id": None})
    assert saved.status_code == 200 and saved.json()["source"] == "user"
    stale = api_client.put(base, json={"text": "x", "expected_revision_id": None})
    assert stale.status_code == 409

    sent = api_client.post(
        f"{base}/messages",
        json={"message_id": "3b9d6f0a-1c2e-4f5a-8b7c-9d0e1f2a3b4c", "text": "More short poems"},
    )
    assert sent.status_code == 200
    assert [item["text"] for item in sent.json()["pending_messages"]] == ["More short poems"]
