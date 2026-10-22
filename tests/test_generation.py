from __future__ import annotations

import json
import shutil
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from threading import Event, Lock
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.agent_brief import build_agent_brief
from server.agent_worker import (
    CallbackInvocation,
    CallbackResult,
    PipelineState,
    _canonicalize_known_term_definitions,
    _freeze_callback_lessons,
    _job_dir,
    _maintain_worker_workspaces,
    _prepare_invocation,
    _reconcile_target_term_keys,
    _refresh_worker_workspaces,
    _round_robin_workspaces,
    _run_lexical_conflict_chunks,
    _same_prose_dependencies,
    _strict_output_schema,
    _validate_generated_lesson_quality,
    process_generation_task,
)
from server.clock import utc_now
from server.content_planning import build_content_plan
from server.db import init_db, session_scope
from server.generation_instructions import (
    LEXICAL_COVERAGE_INSTRUCTIONS,
    _generation_quality_instructions,
    _grammar_generation_instructions,
    _grammar_generation_policy,
)
from server.generation_lexical_stage import TERM_IDENTITY_PREFLIGHT_INSTRUCTIONS
from server.generation_normalization import normalize_callback_lesson
from server.generation_queue import (
    claim_generation_task,
    ensure_generation_task,
    fail_generation_task,
    maintain_generation_task,
    maintain_topic_generation_tasks,
    recover_running_generation_tasks,
    request_topic_lesson,
    retry_generation_task,
)
from server.learning import get_reader_state, import_lesson, record_events
from server.learning_units import generated_unit_errors
from server.models import GenerationTask, Lesson
from server.profile_activation import ProfileActivationUpdate, activate_profile_settings
from server.profile_state import update_profile
from server.schemas import (
    GeneratedLessonDraft,
    GenerationCallbackRequest,
    GenerationCallbackResult,
    GenerationGrammarResult,
    GenerationLexicalBatchRequest,
    GenerationLexicalBatchResult,
    GenerationLexicalConflictRequest,
    GenerationLexicalConflictResult,
    GenerationLexicalResult,
    GenerationLexicalUnitRequest,
    GenerationLexicalUnitResult,
    GenerationProseResult,
    GenerationTargetPolicy,
    GenerationTranslationResult,
    InteractionIn,
    LessonDocument,
    LessonTerm,
    LexicalConflictGroup,
    TextRequestIn,
)
from server.workspaces import WorkspaceRegistry


def _activate_test_profile(db: Session) -> None:
    activate_profile_settings(db, ProfileActivationUpdate())


def test_feedback_validation_progress_and_brief(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    rating = event_factory(
        lesson.id,
        "lesson.rated",
        event_id="rating",
        payload={"rating": 1, "feedback": ["longer", "more_grammar", "same_topic"]},
    )
    record_events(db, [rating])

    progress = get_reader_state(db, lesson_id=lesson.id).progress
    recent = build_agent_brief(db).recent_lessons[0]
    assert progress.feedback == ["longer", "more_grammar", "same_topic"]
    assert recent.feedback == progress.feedback

    duplicate = dict(rating)
    duplicate["event_id"] = "duplicate-feedback"
    duplicate["payload"] = {"rating": 1, "feedback": ["longer", "longer"]}
    with pytest.raises(ValidationError, match="unique"):
        InteractionIn.model_validate(duplicate)

    contradictory = dict(rating)
    contradictory["event_id"] = "contradictory-feedback"
    contradictory["payload"] = {"rating": 1, "feedback": ["shorter", "longer"]}
    with pytest.raises(ValidationError, match="contradictory"):
        InteractionIn.model_validate(contradictory)


def test_feedback_can_be_recorded_without_a_rating(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    feedback = event_factory(
        lesson.id,
        "lesson.rated",
        event_id="feedback-without-rating",
        payload={"rating": None, "feedback": ["shorter", "new_topic"]},
        seconds=1,
    )

    completion = event_factory(
        lesson.id,
        "lesson.completed",
        event_id="completion-without-rating",
        payload={"active_seconds": 60, "completion_ratio": 1},
    )
    record_events(db, [completion, feedback])

    progress = get_reader_state(db, lesson_id=lesson.id).progress
    recent = build_agent_brief(db).recent_lessons[0]
    assert progress.rating is None
    assert progress.feedback == ["shorter", "new_topic"]
    assert recent.rating is None
    assert recent.feedback == progress.feedback

    empty = dict(feedback)
    empty["event_id"] = "empty-feedback-without-rating"
    empty["payload"] = {"rating": None, "feedback": []}
    with pytest.raises(ValidationError, match="rating or feedback"):
        InteractionIn.model_validate(empty)


def test_generation_task_enqueue_dedupe_and_retry(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "1")
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "2")
    lesson = import_lesson(db, lesson_factory())
    complete = event_factory(
        lesson.id,
        "lesson.completed",
        event_id="complete-for-generation",
        payload={"active_seconds": 60, "completion_ratio": 1},
    )
    rating = event_factory(
        lesson.id,
        "lesson.rated",
        event_id="rate-for-generation",
        payload={"rating": -1, "feedback": ["easier"]},
        seconds=1,
    )
    record_events(db, [complete, rating])

    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 1
    task = db.scalar(select(GenerationTask))
    assert task is not None
    assert task.state == "pending"
    assert task.payload["latest_feedback"] == ["easier"]
    assert task.payload["profile_key"] == "es-es-en"
    assert get_reader_state(db).generation_status == "pending"

    claimed = claim_generation_task(db)
    assert claimed is not None
    assert claimed.state == "running"
    failed = fail_generation_task(
        db, claimed.id, log_path=str(tmp_path / "callback.log"), error="test failure"
    )
    assert failed.state == "failed"
    assert get_reader_state(db).generation_status == "failed"
    retried = retry_generation_task(db, failed.id)
    assert retried.state == "pending"
    assert retried.error is None


def test_retry_preserves_a_legacy_failure_that_predates_failure_history(db: Session) -> None:
    update_profile(db, {"level": "A1"})
    task = ensure_generation_task(db, require_enabled=False)
    assert task is not None
    claimed = claim_generation_task(db)
    assert claimed is not None
    claimed.state = "failed"
    claimed.dedupe_key = None
    claimed.error = "legacy validation failure"
    db.commit()

    retried = retry_generation_task(db, claimed.id)

    assert retried.payload["previous_failures"] == [
        {
            "task_id": claimed.id,
            "attempt": 1,
            "error": "legacy validation failure",
        }
    ]
    assert retried.error is None


def test_unrated_completion_does_not_reuse_older_feedback(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "1")
    first = import_lesson(db, lesson_factory(key="first"))
    second = import_lesson(db, lesson_factory(key="second"))
    record_events(
        db,
        [
            event_factory(
                first.id,
                "lesson.completed",
                event_id="older-completion",
                session_id="older-session",
                payload={"active_seconds": 60, "completion_ratio": 1},
            ),
            event_factory(
                first.id,
                "lesson.rated",
                event_id="older-feedback",
                session_id="older-session",
                payload={"rating": -1, "feedback": ["easier"]},
                seconds=1,
            ),
            event_factory(
                second.id,
                "lesson.completed",
                event_id="new-unrated-completion",
                session_id="new-session",
                payload={"active_seconds": 60, "completion_ratio": 1},
                seconds=2,
            ),
        ],
    )

    task = db.scalar(select(GenerationTask))
    assert task is not None
    assert task.payload["latest_feedback"] == []


def test_default_ordinary_queue_stages_one_lesson_per_task(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ARC_LANG_QUEUE_TARGET", raising=False)
    update_profile(db, {"level": "A1"})

    task = ensure_generation_task(db, require_enabled=False)

    assert task is not None
    assert task.payload["queue_target"] == 3
    assert task.payload["queue_shortfall"] == 3
    assert task.payload["needed_lesson_count"] == 1


def test_lesson_start_refills_a_true_ready_ahead_reserve(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "1")
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "3")
    update_profile(db, {"level": "A1"})
    lessons = [import_lesson(db, lesson_factory(key=f"ready-{index}")) for index in range(3)]

    record_events(
        db,
        [event_factory(lessons[0].id, "lesson.started", event_id="start-refill")],
    )

    record_events(
        db,
        [
            event_factory(
                lessons[0].id,
                "term.revealed",
                event_id="reveal-refill",
                payload={"term_key": "es:manana:NOUN"},
            )
        ],
    )
    record_events(
        db,
        [
            event_factory(
                lessons[0].id,
                "translation.revealed",
                event_id="translation-deduped-refill",
                payload={"scope": "lesson"},
                seconds=1,
            )
        ],
    )

    tasks = db.scalars(select(GenerationTask)).all()
    assert len(tasks) == 1
    assert tasks[0].state == "pending"
    assert tasks[0].payload["trigger_event_ids"] == [
        "start-refill",
        "reveal-refill",
        "translation-deduped-refill",
    ]
    assert tasks[0].payload["unread_lesson_count"] == 2
    assert tasks[0].payload["queue_shortfall"] == 1


def test_generation_maintenance_retries_once_then_leaves_failure(
    db: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "3")
    update_profile(db, {"level": "A1"})
    task = ensure_generation_task(db, require_enabled=False)
    assert task is not None
    claimed = claim_generation_task(db)
    assert claimed is not None
    fail_generation_task(
        db,
        claimed.id,
        log_path=str(tmp_path / "attempt-1.log"),
        error="first failure",
    )

    retried = maintain_generation_task(db)

    assert retried is not None
    assert retried.id == task.id
    assert retried.state == "pending"
    second = claim_generation_task(db)
    assert second is not None
    assert second.attempts == 2
    fail_generation_task(
        db,
        second.id,
        log_path=str(tmp_path / "attempt-2.log"),
        error="second failure",
    )

    stopped = maintain_generation_task(db)

    assert stopped is not None
    assert stopped.id == task.id
    assert stopped.state == "failed"
    assert stopped.attempts == 2
    assert [item["error"] for item in stopped.payload["previous_failures"]] == [
        "first failure",
        "second failure",
    ]
    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 1

    legacy_payload = dict(stopped.payload)
    legacy_payload["generation_contract_revision"] = 1
    stopped.payload = legacy_payload
    db.commit()

    replacement = maintain_generation_task(db)

    assert replacement is not None
    assert replacement.id != stopped.id
    assert replacement.state == "pending"
    assert replacement.payload["previous_failures"] == stopped.payload["previous_failures"]


def test_fresh_event_recovers_terminal_failure_once_and_reaches_retry_request(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "1")
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "3")
    update_profile(db, {"level": "A1"})
    lesson = import_lesson(db, lesson_factory())
    original = ensure_generation_task(db, require_enabled=False)
    assert original is not None

    first = claim_generation_task(db)
    assert first is not None
    fail_generation_task(
        db,
        first.id,
        log_path=str(tmp_path / "first.log"),
        error="invalid learning units: split 这个",
    )
    maintain_generation_task(db)
    second = claim_generation_task(db)
    assert second is not None
    fail_generation_task(
        db,
        second.id,
        log_path=str(tmp_path / "second.log"),
        error="library identity conflict for zh:test:WORD",
    )
    stopped = maintain_generation_task(db)
    assert stopped is not None and stopped.id == original.id

    result = record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="fresh-after-terminal",
                payload={"term_key": "es:manana:NOUN"},
            )
        ],
    )

    assert result.accepted == 1
    replacement = db.scalar(select(GenerationTask).order_by(GenerationTask.id.desc()))
    assert replacement is not None
    assert replacement.id != original.id
    assert replacement.payload["trigger_event_ids"] == ["fresh-after-terminal"]
    assert [item["error"] for item in replacement.payload["previous_failures"]] == [
        "invalid learning units: split 这个",
        "library identity conflict for zh:test:WORD",
    ]

    claimed = claim_generation_task(db)
    assert claimed is not None
    fail_generation_task(
        db,
        claimed.id,
        log_path=str(tmp_path / "replacement-first.log"),
        error="replacement failure one",
    )
    maintain_generation_task(db)
    final_attempt = claim_generation_task(db)
    assert final_attempt is not None
    fail_generation_task(
        db,
        final_attempt.id,
        log_path=str(tmp_path / "replacement-second.log"),
        error="replacement failure two",
    )

    replay = ensure_generation_task(
        db,
        trigger_event_ids=["fresh-after-terminal"],
        require_enabled=False,
    )
    maintained = maintain_generation_task(db)

    assert replay is not None and replay.id == replacement.id and replay.state == "failed"
    assert maintained is not None and maintained.id == replacement.id
    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 2


def test_retry_request_exposes_exact_previous_validation_failure(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1"})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None
        fail_generation_task(
            db,
            claimed.id,
            log_path=str(tmp_path / "first.log"),
            error="invalid learning units: split 这个",
        )
        retry_generation_task(db, claimed.id)
        retry = claim_generation_task(db)
        assert retry is not None

    _, invocation = _prepare_invocation(workspace, retry)
    try:
        assert [item.error for item in invocation.request.previous_failures] == [
            "invalid learning units: split 这个"
        ]
        assert "validation-aware retry" in invocation.request.instructions
        assert "split 这个" in invocation.request.instructions
        assert invocation.request.stage == "prose"
        assert TERM_IDENTITY_PREFLIGHT_INSTRUCTIONS not in invocation.request.instructions
    finally:
        invocation.job_dir.rmdir()


def test_retry_request_carries_the_rejected_compact_draft_for_in_place_repair(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceRegistry(tmp_path).create(
        "zh-hans",
        learning_language="zh-Hans",
        translation_language="en",
    )
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        first = claim_generation_task(db)
        assert first is not None

    wrong = _term_run("湖", "shui-water", gloss="water")
    wrong["term"].update({"lemma": "水", "pos": "noun", "frequency_rank": 200})
    payload = _chinese_generated_draft([])
    payload["blocks"][0]["sentences"] = [
        {
            "key": f"sentence-{index}",
            "runs": [deepcopy(wrong) for _ in range(100)],
            "translation": "A lake.",
        }
        for index in range(1, 4)
    ]
    response = GenerationCallbackResult.model_validate({"schema_version": 1, "lessons": [payload]})

    class FixedCallback:
        def run(self, _invocation: CallbackInvocation) -> CallbackResult:
            return CallbackResult(exit_code=0, payload=response.model_dump_json(), log="ok")

    process_generation_task(workspace, first, FixedCallback())

    with session_scope(workspace) as db:
        failed = db.get(GenerationTask, first.id)
        assert failed is not None
        assert failed.state == "failed"
        assert "surface text must equal" in (failed.error or "")
        retry_generation_task(db, first.id)
        retry = claim_generation_task(db)
        assert retry is not None
        assert retry.attempts == 2

    _, invocation = _prepare_invocation(workspace, retry)
    try:
        assert len(invocation.request.repair_lessons) == 1
        repair_run = invocation.request.repair_lessons[0].blocks[0].sentences[0].runs[0]
        assert (repair_run.text, repair_run.term_key) == ("湖", "shui-water")
        assert "Repair those drafts in place" in invocation.request.instructions
        assert "Do not write a replacement story" in invocation.request.instructions
        prior_response = (
            workspace.directory / "agent" / "jobs" / f"task-{first.id}-attempt-1/response.json"
        )
        assert prior_response.is_file()
        assert GenerationCallbackResult.model_validate_json(prior_response.read_text())
    finally:
        invocation.job_dir.rmdir()


def test_pending_triggers_are_consumed_and_running_triggers_are_deferred(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "1")
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "3")
    update_profile(db, {"level": "A1"})
    lesson = import_lesson(db, lesson_factory())
    task = ensure_generation_task(db, require_enabled=False)
    assert task is not None

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="while-pending",
                payload={"term_key": "es:manana:NOUN"},
            )
        ],
    )
    db.refresh(task)
    assert task.payload["trigger_event_ids"] == ["while-pending"]

    first = claim_generation_task(db)
    assert first is not None
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "translation.revealed",
                event_id="during-first-attempt",
                payload={"scope": "lesson"},
                seconds=1,
            )
        ],
    )
    db.refresh(first)
    assert first.payload["deferred_trigger_event_ids"] == ["during-first-attempt"]
    fail_generation_task(db, first.id, log_path=str(tmp_path / "first.log"), error="first")

    retried = maintain_generation_task(db)
    assert retried is not None
    assert retried.payload["trigger_event_ids"] == [
        "while-pending",
        "during-first-attempt",
    ]
    assert retried.payload["deferred_trigger_event_ids"] == []

    second = claim_generation_task(db)
    assert second is not None
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="during-terminal-attempt",
                payload={"term_key": "es:manana:NOUN"},
                seconds=2,
            )
        ],
    )
    fail_generation_task(db, second.id, log_path=str(tmp_path / "second.log"), error="second")

    successor = maintain_generation_task(db)

    assert successor is not None and successor.id != task.id
    assert successor.payload["trigger_event_ids"] == [
        "while-pending",
        "during-first-attempt",
        "during-terminal-attempt",
    ]
    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 2


def test_maintenance_recovers_generation_trigger_when_enqueue_wakeup_was_missed(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "0")
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "3")
    update_profile(db, {"level": "A1"})
    lesson = import_lesson(db, lesson_factory())
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

    record_events(
        db,
        [event_factory(lesson.id, "lesson.started", event_id="relevant-start-wakeup")],
    )
    replacement = maintain_generation_task(db)
    assert replacement is not None and replacement.id != original.id
    assert replacement.payload["trigger_event_ids"] == ["relevant-start-wakeup"]
    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 2

    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="relevant-missed-wakeup",
                payload={"term_key": "es:manana:NOUN"},
                seconds=1,
            )
        ],
    )
    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 2

    maintained = maintain_generation_task(db)

    assert maintained is not None and maintained.id == replacement.id
    assert maintained.payload["trigger_event_ids"] == [
        "relevant-start-wakeup",
        "relevant-missed-wakeup",
    ]
    assert maintained.payload["trigger_event_cursor"] > original.payload["trigger_event_cursor"]
    assert db.scalar(select(func.count()).select_from(GenerationTask)) == 2


def test_retry_request_refreshes_and_persists_latest_feedback_snapshot(
    tmp_path: Path,
    lesson_factory: Any,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "0")
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1"})
        lesson = import_lesson(db, lesson_factory())
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        first = claim_generation_task(db)
        assert first is not None

    _, first_invocation = _prepare_invocation(workspace, first)
    try:
        assert first_invocation.request.latest_feedback == []
    finally:
        first_invocation.job_dir.rmdir()

    with session_scope(workspace) as db:
        fail_generation_task(
            db,
            task.id,
            log_path=str(tmp_path / "first.log"),
            error="first validation failure",
        )
        record_events(
            db,
            [
                event_factory(
                    lesson.id,
                    "lesson.completed",
                    event_id="feedback-completion",
                    payload={"active_seconds": 60, "completion_ratio": 1},
                ),
                event_factory(
                    lesson.id,
                    "lesson.rated",
                    event_id="feedback-rating",
                    payload={"rating": -1, "feedback": ["easier", "new_topic"]},
                    seconds=1,
                ),
            ],
        )
        retry_generation_task(db, task.id)
        retry = claim_generation_task(db)
        assert retry is not None

    _, retry_invocation = _prepare_invocation(workspace, retry)
    try:
        assert retry_invocation.request.latest_feedback == ["easier", "new_topic"]
    finally:
        retry_invocation.job_dir.rmdir()

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, task.id)
        assert stored is not None
        assert stored.payload["latest_feedback"] == ["easier", "new_topic"]


def _routing_test_request(
    db: Session,
    lesson_factory: Any,
) -> GenerationCallbackRequest:
    import_lesson(db, lesson_factory())
    brief = build_agent_brief(db)
    return GenerationCallbackRequest(
        job_id="es-es-en:1",
        task_id=1,
        attempt=1,
        lesson_count=1,
        profile_key="es-es-en",
        workspace_path="profiles/es-es-en",
        profile_fingerprint="a" * 64,
        brief=brief,
        target_policy=GenerationTargetPolicy(
            preferred_count=0,
            max_count=0,
            target_text_length=300,
            candidate_term_keys=[],
        ),
        content_plan=build_content_plan(
            brief,
            task_id=1,
            requested_topic=None,
        ),
        instructions="Generate one lesson.",
    )


def test_prose_reuse_depends_on_the_adaptive_snapshot(
    db: Session,
    lesson_factory: Any,
) -> None:
    request = _routing_test_request(db, lesson_factory)
    retry_metadata_only = request.model_copy(
        update={
            "attempt": 2,
            "instructions": "Retry diagnostics changed.",
        }
    )
    assert _same_prose_dependencies(request, retry_metadata_only)

    changed_feedback = request.model_copy(update={"latest_feedback": ["easier"]})
    assert not _same_prose_dependencies(request, changed_feedback)

    changed_target = request.model_copy(
        update={
            "target_policy": request.target_policy.model_copy(
                update={"target_text_length": request.target_policy.target_text_length + 1}
            )
        }
    )
    assert not _same_prose_dependencies(request, changed_target)

    changed_grammar = request.model_copy(
        update={
            "grammar_policy": request.grammar_policy.model_copy(
                update={"max_count": 1, "offered_keys": ("test:grammar",)}
            )
        }
    )
    assert not _same_prose_dependencies(request, changed_grammar)


def _nested_object_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_nested_object_keys(item))
        return keys
    if isinstance(value, list):
        list_keys: set[str] = set()
        for item in value:
            list_keys.update(_nested_object_keys(item))
        return list_keys
    return set()


def test_strict_output_schema_removes_annotations_beside_references() -> None:
    schema = _strict_output_schema(GenerationCallbackResult.model_json_schema())
    callback_draft = schema["$defs"]["CallbackLessonDraft"]
    title_sentence = callback_draft["properties"]["title_sentence"]
    generated_sentences = schema["$defs"]["CallbackLessonBlock"]["properties"]["sentences"]
    callback_run = schema["$defs"]["CallbackLessonRun"]["properties"]

    assert set(title_sentence) == {"$ref"}
    assert generated_sentences["maxItems"] == 20
    assert "terms" in callback_draft["properties"]
    assert "term_key" in callback_run
    assert "term" not in callback_run
    assert "GeneratedLessonDraft" not in schema["$defs"]

    pending: list[Any] = [schema]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            assert "$ref" not in value or len(value) == 1
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def test_prose_output_schema_contains_only_frozen_text_structure() -> None:
    schema = _strict_output_schema(GenerationProseResult.model_json_schema())
    draft = schema["$defs"]["ProseLessonDraft"]["properties"]
    sentence = schema["$defs"]["ProseLessonSentence"]["properties"]

    assert set(sentence) == {"key", "text"}
    assert "terms" not in draft
    assert "translation" not in json.dumps(schema)
    assert "grammar" not in json.dumps(schema)


def test_split_stage_output_schemas_are_compact_and_task_specific() -> None:
    lexical = _strict_output_schema(GenerationLexicalResult.model_json_schema())
    translation = _strict_output_schema(GenerationTranslationResult.model_json_schema())
    grammar = _strict_output_schema(GenerationGrammarResult.model_json_schema())

    lexical_sentence = lexical["$defs"]["LexicalLessonSentence"]["properties"]
    assert set(lexical_sentence) == {"key", "runs"}
    assert "translation" not in lexical_sentence
    assert "grammar" not in lexical_sentence

    translation_sentence = translation["$defs"]["SentenceTranslationDraft"]["properties"]
    assert set(translation_sentence) == {"key", "translation"}
    assert "runs" not in json.dumps(translation)
    assert "terms" not in json.dumps(translation)

    grammar_occurrence = grammar["$defs"]["CallbackGrammarOccurrence"]["properties"]
    assert set(grammar_occurrence) == {
        "construction_key",
        "run_start",
        "run_end",
        "note",
    }
    assert "key" not in grammar_occurrence
    assert "translation" not in json.dumps(grammar)


def test_strict_output_schema_rejects_structural_reference_siblings() -> None:
    with pytest.raises(ValueError, match="structural siblings: minLength"):
        _strict_output_schema({"$ref": "#/$defs/Text", "minLength": 1})


def test_worker_refreshes_new_profiles_and_round_robins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server import agent_worker

    workspace_registry = WorkspaceRegistry(tmp_path)
    monkeypatch.setattr(agent_worker, "registry", workspace_registry)
    known_profiles: set[str] = set()

    initial = _refresh_worker_workspaces(None, known_profiles)
    assert [workspace.profile_id for workspace in initial] == ["es-es"]
    workspace_registry.create("profile-b", learning_language="fr", translation_language="en")
    refreshed = _refresh_worker_workspaces(None, known_profiles)

    assert [workspace.profile_id for workspace in refreshed] == ["es-es", "profile-b"]
    assert (tmp_path / "profiles" / "profile-b" / "lang.db").is_file()
    assert [workspace.profile_id for workspace in _round_robin_workspaces(refreshed, "es-es")] == [
        "profile-b",
        "es-es",
    ]
    assert [
        workspace.profile_id
        for workspace in _round_robin_workspaces(
            refreshed,
            "es-es",
            preferred_profile_id="es-es",
        )
    ] == ["es-es", "profile-b"]


def test_worker_artifact_directories_are_private(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    task = GenerationTask(id=1, attempts=1)

    job_directory = _job_dir(workspace, task)

    for directory in (
        workspace.directory / "agent",
        workspace.directory / "agent" / "jobs",
        job_directory,
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def _term_run(
    text: str, key: str, *, gloss: str | None = None, frequency_rank: int = 100
) -> dict[str, Any]:
    return {
        "text": text,
        "term": {
            "key": key,
            "lemma": text.casefold(),
            "pos": "WORD",
            "gloss": gloss or f"gloss for {text}",
            "frequency_rank": frequency_rank,
        },
    }


def _generated_draft(
    runs: list[dict[str, Any]],
    *,
    targets: list[str] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "title": "Prueba",
        "title_sentence": {
            "key": "title-sentence",
            "runs": [_term_run("Prueba", "es:prueba:NOUN")],
            "translation": "Test",
        },
        "topic": "testing",
        "subject_domain": "music",
        "level": "A1",
        "difficulty": 0.1,
        "blocks": [
            {
                "key": "block-1",
                "sentences": [
                    {
                        "key": "sentence-1",
                        "runs": runs,
                        "translation": "Coverage test translation.",
                    }
                ],
            }
        ],
        "target_term_keys": targets or [],
        "calibration": calibration,
    }


def _callback_sized_generated_draft(
    runs: list[dict[str, Any]],
    *,
    target_length: int = 300,
    difficulty: float = 0.1,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    lexical_runs = [*runs]
    lexical_runs.extend(
        [_term_run("relleno", "es:relleno:WORD")] * (target_length - len(lexical_runs))
    )
    payload = _generated_draft([], targets=targets)
    payload["difficulty"] = difficulty
    payload["blocks"][0]["sentences"] = [
        {
            "key": f"sentence-{index}",
            "runs": lexical_runs[offset : offset + 100],
            "translation": "Callback quality fixture.",
        }
        for index, offset in enumerate(range(0, len(lexical_runs), 100), start=1)
    ]
    return payload


def _staged_callback_fixture(
    *,
    target_length: int = 300,
) -> tuple[
    GenerationProseResult,
    GenerationLexicalResult,
    GenerationTranslationResult,
    GenerationGrammarResult,
    GenerationCallbackResult,
]:
    response = GenerationCallbackResult.model_validate(
        {
            "schema_version": 1,
            "lessons": [
                _callback_sized_generated_draft(
                    [_term_run("hola", "es:hola:WORD")],
                    target_length=target_length,
                    difficulty=0.1,
                )
            ],
        }
    )
    prose = GenerationProseResult(
        schema_version=1,
        lessons=_freeze_callback_lessons(response.lessons),
    )
    lexical_lessons: list[dict[str, Any]] = []
    translation_lessons: list[dict[str, Any]] = []
    grammar_lessons: list[dict[str, Any]] = []
    for lesson in response.lessons:
        lexical_lessons.append(
            {
                "terms": lesson.terms,
                "title_sentence": {
                    "key": lesson.title_sentence.key,
                    "runs": lesson.title_sentence.runs,
                },
                "blocks": [
                    {
                        "key": block.key,
                        "sentences": [
                            {"key": sentence.key, "runs": sentence.runs}
                            for sentence in block.sentences
                        ],
                    }
                    for block in lesson.blocks
                ],
                "target_term_keys": lesson.target_term_keys,
            }
        )
        sentences = [
            lesson.title_sentence,
            *(sentence for block in lesson.blocks for sentence in block.sentences),
        ]
        translation_lessons.append(
            {
                "sentences": [
                    {"key": sentence.key, "translation": sentence.translation}
                    for sentence in sentences
                ]
            }
        )
        grammar_lessons.append(
            {
                "sentences": [
                    {
                        "key": sentence.key,
                        "occurrences": [
                            {
                                "construction_key": occurrence.construction_key,
                                "run_start": occurrence.run_start,
                                "run_end": occurrence.run_end,
                                "note": occurrence.note,
                            }
                            for occurrence in sentence.grammar
                        ],
                    }
                    for sentence in sentences
                ]
            }
        )
    lexical = GenerationLexicalResult.model_validate(
        {"schema_version": 1, "lessons": lexical_lessons}
    )
    translation = GenerationTranslationResult.model_validate(
        {"schema_version": 1, "lessons": translation_lessons}
    )
    grammar = GenerationGrammarResult.model_validate(
        {"schema_version": 1, "lessons": grammar_lessons}
    )
    return prose, lexical, translation, grammar, response


def _lexical_unit_payload(
    payloads: dict[str, str], lesson_index: int, sentence_index: int, unit_id: str
) -> GenerationLexicalUnitResult:
    aggregate = GenerationLexicalResult.model_validate_json(payloads["lexical"])
    lesson = aggregate.lessons[lesson_index]
    sentences = [
        lesson.title_sentence,
        *(sentence for block in lesson.blocks for sentence in block.sentences),
    ]
    sentence = sentences[sentence_index]
    referenced = {run.term_key for run in sentence.runs if run.term_key is not None}
    return GenerationLexicalUnitResult(
        schema_version=1,
        unit_id=unit_id,
        key=sentence.key,
        terms=[term for term in lesson.terms if term.key in referenced],
        runs=sentence.runs,
    )


def _staged_payload(
    invocation: CallbackInvocation,
    payloads: dict[str, str],
) -> str:
    request = invocation.request
    if isinstance(request, GenerationLexicalUnitRequest):
        return _lexical_unit_payload(
            payloads, request.lesson_index, request.sentence_index, request.unit_id
        ).model_dump_json()
    if isinstance(request, GenerationLexicalBatchRequest):
        return GenerationLexicalBatchResult(
            schema_version=1,
            units=[
                _lexical_unit_payload(
                    payloads, unit.lesson_index, unit.sentence_index, unit.unit_id
                )
                for unit in request.units
            ],
        ).model_dump_json()
    if isinstance(request, GenerationLexicalConflictRequest):
        return GenerationLexicalConflictResult(
            schema_version=1,
            resolutions=[
                {
                    "candidate_id": candidate.candidate_id,
                    "canonical_candidate_id": candidate.candidate_id,
                }
                for group in request.groups
                for candidate in group.candidates
                if not candidate.stored
            ],
        ).model_dump_json()
    return payloads[request.stage]


def _source_drifted_lexical_payload(
    invocation: CallbackInvocation,
    payloads: dict[str, str],
) -> str:
    payload = json.loads(_staged_payload(invocation, payloads))
    payload["runs"][0]["text"] += "x"
    return json.dumps(payload, ensure_ascii=False)


def test_worker_repairs_only_the_invalid_lexical_unit_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercises the per-sentence path that batching falls back to.
    monkeypatch.setenv("ARC_LANG_LEXICAL_BATCH_SIZE", "1")
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }
    calls: dict[str, int] = {}
    lexical_requests: list[GenerationLexicalUnitRequest] = []
    lock = Lock()

    class RepairingCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            request = invocation.request
            if isinstance(request, GenerationLexicalUnitRequest):
                with lock:
                    calls[request.unit_id] = calls.get(request.unit_id, 0) + 1
                    lexical_requests.append(request)
                if request.frozen_sentence.key == "sentence-2" and request.unit_attempt == 1:
                    payload = _source_drifted_lexical_payload(invocation, payloads)
                else:
                    payload = _staged_payload(invocation, payloads)
            else:
                payload = _staged_payload(invocation, payloads)
            return CallbackResult(exit_code=0, payload=payload, log=request.stage)

    process_generation_task(workspace, claimed, RepairingCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
    target_requests = [
        request for request in lexical_requests if request.frozen_sentence.key == "sentence-2"
    ]
    assert [request.unit_attempt for request in target_requests] == [1, 2]
    target_id = target_requests[0].unit_id
    assert calls[target_id] == 2
    assert all(count == 1 for unit_id, count in calls.items() if unit_id != target_id)
    repair = target_requests[1]
    assert repair.repair_result is not None
    assert repair.repair_result.unit_id == repair.unit_id
    assert repair.repair_result.key == repair.frozen_sentence.key
    assert "".join(run.text for run in repair.repair_result.runs) != repair.frozen_sentence.text
    assert "did not reconstruct frozen source exactly" in repair.instructions
    assert sum(request.unit_attempt == 2 for request in lexical_requests) == 1


def _claimed_staged_task(tmp_path: Path) -> tuple[Any, GenerationTask, dict[str, str]]:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        assert ensure_generation_task(db, require_enabled=False) is not None
        claimed = claim_generation_task(db)
        assert claimed is not None
    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }
    return workspace, claimed, payloads


def test_batched_lexical_units_fall_back_per_sentence(tmp_path: Path) -> None:
    workspace, claimed, payloads = _claimed_staged_task(tmp_path)
    singles: list[GenerationLexicalUnitRequest] = []
    batches: list[GenerationLexicalBatchRequest] = []
    lock = Lock()

    class PartlyWrongBatch:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            request = invocation.request
            payload = _staged_payload(invocation, payloads)
            if isinstance(request, GenerationLexicalBatchRequest):
                with lock:
                    batches.append(request)
                data = json.loads(payload)
                kept = []
                for unit in data["units"]:
                    if unit["key"] == "sentence-1":
                        continue  # omitted unit -> fresh single-sentence attempt
                    if unit["key"] == "sentence-2":
                        unit["runs"][0]["text"] += "x"  # drifted unit -> local repair
                    kept.append(unit)
                data["units"] = kept
                payload = json.dumps(data, ensure_ascii=False)
            elif isinstance(request, GenerationLexicalUnitRequest):
                with lock:
                    singles.append(request)
            return CallbackResult(exit_code=0, payload=payload, log=request.stage)

    process_generation_task(workspace, claimed, PartlyWrongBatch())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
    assert len(batches) == 1 and len(batches[0].units) == 4
    by_key = {request.frozen_sentence.key: request for request in singles}
    assert set(by_key) == {"sentence-1", "sentence-2"}
    assert by_key["sentence-1"].unit_attempt == 1 and by_key["sentence-1"].repair_result is None
    repair = by_key["sentence-2"]
    assert repair.unit_attempt == 2 and repair.repair_result is not None
    assert "did not reconstruct frozen source exactly" in repair.instructions


def test_failed_lexical_batch_retries_every_sentence_singly(tmp_path: Path) -> None:
    workspace, claimed, payloads = _claimed_staged_task(tmp_path)
    singles: list[str] = []
    lock = Lock()

    class FailingBatch:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            request = invocation.request
            if isinstance(request, GenerationLexicalBatchRequest):
                return CallbackResult(exit_code=1, payload="", log="batch failed")
            if isinstance(request, GenerationLexicalUnitRequest):
                with lock:
                    singles.append(request.frozen_sentence.key)
            return CallbackResult(
                exit_code=0, payload=_staged_payload(invocation, payloads), log=request.stage
            )

    process_generation_task(workspace, claimed, FailingBatch())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
    assert sorted(singles) == ["sentence-1", "sentence-2", "sentence-3", "title-sentence"]


def test_task_retry_reuses_successful_partial_lexical_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercises the per-sentence path that batching falls back to.
    monkeypatch.setenv("ARC_LANG_LEXICAL_BATCH_SIZE", "1")
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        first = claim_generation_task(db)
        assert first is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }
    first_calls: dict[str, int] = {}
    lock = Lock()

    class ExhaustingCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            request = invocation.request
            if isinstance(request, GenerationLexicalUnitRequest):
                with lock:
                    first_calls[request.unit_id] = first_calls.get(request.unit_id, 0) + 1
                if request.frozen_sentence.key == "sentence-2":
                    payload = _source_drifted_lexical_payload(invocation, payloads)
                else:
                    payload = _staged_payload(invocation, payloads)
            else:
                payload = _staged_payload(invocation, payloads)
            return CallbackResult(exit_code=0, payload=payload, log=request.stage)

    process_generation_task(workspace, first, ExhaustingCallback())

    with session_scope(workspace) as db:
        failed = db.get(GenerationTask, first.id)
        assert failed is not None
        assert failed.state == "failed"
        assert (failed.error or "").startswith("[lexical]")
        retry_generation_task(db, failed.id)
        second = claim_generation_task(db)
        assert second is not None and second.attempts == 2
    assert sorted(first_calls.values()) == [1, 1, 1, 2]

    retry_requests: list[Any] = []

    class SuccessfulTaskRetry:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            retry_requests.append(invocation.request)
            return CallbackResult(
                exit_code=0,
                payload=_staged_payload(invocation, payloads),
                log=invocation.request.stage,
            )

    process_generation_task(workspace, second, SuccessfulTaskRetry())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, second.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
        assert db.scalar(select(func.count()).select_from(Lesson)) == 1
    assert len(retry_requests) == 2
    retried_unit = retry_requests[0]
    assert isinstance(retried_unit, GenerationLexicalUnitRequest)
    assert retried_unit.frozen_sentence.key == "sentence-2"
    assert retried_unit.unit_attempt == 1
    assert retried_unit.repair_result is not None
    assert retry_requests[1].stage == "grammar"


def test_callback_concurrency_is_bounded_across_translation_and_lexical_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARC_LANG_LEXICAL_BATCH_SIZE", "1")
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }
    lock = Lock()
    three_active = Event()
    hold = Event()
    active_total = 0
    active_lexical = 0
    max_total = 0
    max_lexical = 0

    class ObservedCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            nonlocal active_total, active_lexical, max_total, max_lexical
            request = invocation.request
            is_lexical = isinstance(request, GenerationLexicalUnitRequest)
            is_parallel_stage = is_lexical or request.stage == "translation"
            with lock:
                active_total += 1
                active_lexical += int(is_lexical)
                max_total = max(max_total, active_total)
                max_lexical = max(max_lexical, active_lexical)
                if active_total == 3 and active_lexical == 2:
                    three_active.set()
            try:
                if is_parallel_stage:
                    assert three_active.wait(1)
                    hold.wait(0.05)
                return CallbackResult(
                    exit_code=0,
                    payload=_staged_payload(invocation, payloads),
                    log=request.stage,
                )
            finally:
                with lock:
                    active_total -= 1
                    active_lexical -= int(is_lexical)

    process_generation_task(workspace, claimed, ObservedCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
    assert three_active.is_set()
    assert max_total == 3
    assert max_lexical == 2


def test_lexical_conflicts_over_schema_limit_are_split_and_recombined(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    log_path, initial = _prepare_invocation(workspace, claimed)
    base_request = initial.base_request
    assert base_request is not None
    groups = [
        LexicalConflictGroup.model_validate(
            {
                "group_id": f"group-{index}",
                "candidates": [
                    {
                        "candidate_id": f"stored-{index}",
                        "term": {
                            "key": f"stored-key-{index}",
                            "lemma": f"lemma-{index}",
                            "pos": "noun",
                            "gloss": "stored",
                            "pronunciation": None,
                            "frequency_rank": 100 + index,
                        },
                        "stored": True,
                        "contexts": [],
                    },
                    {
                        "candidate_id": f"generated-{index}",
                        "term": {
                            "key": f"local-{index}",
                            "lemma": f"lemma-{index}",
                            "pos": "noun",
                            "gloss": "generated",
                            "pronunciation": None,
                            "frequency_rank": 100 + index,
                        },
                        "stored": False,
                        "contexts": [
                            {
                                "sentence_key": f"sentence-{index}",
                                "surface": f"lemma-{index}",
                                "sentence_text": f"lemma-{index}.",
                            }
                        ],
                    },
                ],
            }
        )
        for index in range(65)
    ]
    request_sizes: list[int] = []
    request_lock = Lock()

    class ConflictCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            request = invocation.request
            assert isinstance(request, GenerationLexicalConflictRequest)
            with request_lock:
                request_sizes.append(len(request.groups))
            result = GenerationLexicalConflictResult(
                schema_version=1,
                resolutions=[
                    {
                        "candidate_id": candidate.candidate_id,
                        "canonical_candidate_id": candidate.candidate_id,
                    }
                    for group in request.groups
                    for candidate in group.candidates
                    if not candidate.stored
                ],
            )
            return CallbackResult(exit_code=0, payload=result.model_dump_json(), log="ok")

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = _run_lexical_conflict_chunks(
                ConflictCallback(),
                executor,
                base_request,
                groups,
                job_dir=initial.job_dir,
                artifact_dir=log_path.parent,
                state=PipelineState(),
                log_path=log_path,
                logs=[],
            )
    finally:
        shutil.rmtree(initial.job_dir, ignore_errors=True)

    assert sorted(request_sizes) == [1, 64]
    assert len(results) == 2
    events = [
        json.loads(line) for line in (log_path.parent / "timings.jsonl").read_text().splitlines()
    ]
    assert sorted(event["conflict_groups"] for event in events if event["kind"] == "callback") == [
        1,
        64,
    ]


def test_worker_runs_the_task_dag_with_separate_artifacts(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }
    observed: list[Any] = []
    lock = Lock()

    class StagedCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            with lock:
                observed.append(invocation.request)
            return CallbackResult(
                exit_code=0,
                payload=_staged_payload(invocation, payloads),
                log=invocation.request.stage,
            )

    process_generation_task(workspace, claimed, StagedCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
        assert db.scalar(select(func.count()).select_from(Lesson)) == 1
        imported = db.scalar(select(Lesson))
        assert imported is not None
        imported_document = LessonDocument.model_validate(imported.payload)
    prose_request = observed[0]
    assert prose_request.stage == "prose"
    assert {(item.stage, item.task) for item in observed} == {
        ("prose", "prose"),
        ("lexical", "lexical"),
        ("translation", "translation"),
        ("grammar", "grammar"),
    }
    lexical_requests = [
        item for item in observed if isinstance(item, GenerationLexicalBatchRequest)
    ]
    translation_request = next(item for item in observed if item.stage == "translation")
    grammar_request = next(item for item in observed if item.stage == "grammar")
    assert "Final prose preflight" in prose_request.instructions
    assert "every content_plan field" in prose_request.instructions
    assert prose_request.grammar_policy.offered_keys
    assert (
        f"The adaptive offered pool is "
        f"{json.dumps(prose_request.grammar_policy.offered_keys, ensure_ascii=False)}."
        in prose_request.instructions
    )
    assert imported_document.metadata["grammar"]["preferred"] == (
        prose_request.grammar_policy.preferred_count
    )
    assert imported_document.metadata["grammar"]["max"] == prose_request.grammar_policy.max_count
    assert imported_document.metadata["grammar"]["offered"] == list(
        prose_request.grammar_policy.offered_keys
    )
    assert len(lexical_requests) == 1
    assert [unit.frozen_sentence.key for unit in lexical_requests[0].units] == [
        "title-sentence",
        "sentence-1",
        "sentence-2",
        "sentence-3",
    ]
    assert all("local to that one unit" in request.instructions for request in lexical_requests)
    assert all("never translate" in request.instructions for request in lexical_requests)
    assert translation_request.frozen_lessons == prose.lessons
    assert grammar_request.tokenized_lessons
    artifact_dir = (
        workspace.directory / "agent" / "jobs" / f"task-{claimed.id}-attempt-{claimed.attempts}"
    )
    for stage in ("prose", "lexical", "translation", "grammar"):
        assert (artifact_dir / f"{stage}-request.json").is_file()
        assert (artifact_dir / f"{stage}-response-schema.json").is_file()
        assert (artifact_dir / f"{stage}-response.json").is_file()
        assert (artifact_dir / f"{stage}-callback.log").is_file()
        if stage != "lexical":
            assert (
                (artifact_dir / f"{stage}-callback.log")
                .read_text()
                .startswith("timing: duration_seconds=")
            )
    assert len(list(artifact_dir.glob("lexical-batch-*-request.json"))) == 1
    assert len(list(artifact_dir.glob("lexical-batch-*-response.json"))) == 1
    timing_events = [
        json.loads(line) for line in (artifact_dir / "timings.jsonl").read_text().splitlines()
    ]
    callback_events = [event for event in timing_events if event["kind"] == "callback"]
    assert {event["artifact_stem"] for event in callback_events} == {
        "prose",
        "translation",
        "grammar",
        "lexical-batch-00",
    }
    assert all(event["duration_seconds"] >= 0 for event in callback_events)
    assert all(event["queue_wait_seconds"] >= 0 for event in callback_events)
    assert all(event["request_bytes"] > 0 for event in callback_events)
    assert {event["operation"] for event in timing_events if event["kind"] == "host"} >= {
        "prepare",
        "lexical_stage",
        "assembly",
        "import",
        "task",
    }
    grammar_schema = json.loads(
        (artifact_dir / "grammar-response-schema.json").read_text(encoding="utf-8")
    )
    assert grammar_schema["$defs"]["CallbackGrammarOccurrence"]["properties"]["construction_key"][
        "enum"
    ] == [entry.key for entry in grammar_request.grammar_catalog]


def test_lexical_and_translation_callbacks_start_concurrently(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }
    lexical_started = Event()
    translation_started = Event()

    class ConcurrentCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            if invocation.request.stage == "lexical":
                lexical_started.set()
                assert translation_started.wait(2)
            elif invocation.request.stage == "translation":
                translation_started.set()
                assert lexical_started.wait(2)
            return CallbackResult(
                exit_code=0,
                payload=_staged_payload(invocation, payloads),
                log=invocation.request.stage,
            )

    process_generation_task(workspace, claimed, ConcurrentCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
    assert lexical_started.is_set()
    assert translation_started.is_set()


def test_worker_rejects_lexical_output_that_changes_frozen_text(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    changed_wire = lexical.model_dump(mode="json")
    changed_wire["lessons"][0]["blocks"][0]["sentences"][0]["runs"][0]["text"] = "adiós"
    changed = GenerationLexicalResult.model_validate(changed_wire)
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": changed.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }

    class TextChangingCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            return CallbackResult(
                exit_code=0,
                payload=_staged_payload(invocation, payloads),
                log=invocation.request.stage,
            )

    process_generation_task(workspace, claimed, TextChangingCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "failed"
        assert "[lexical]" in (stored.error or "")
        assert "did not reconstruct frozen source exactly" in (stored.error or "")
        assert db.scalar(select(func.count()).select_from(Lesson)) == 0


def test_validated_short_lexical_output_invalidates_prose_on_retry(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        first = claim_generation_task(db)
        assert first is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture(target_length=128)
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }

    class ShortCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            return CallbackResult(
                exit_code=0,
                payload=_staged_payload(invocation, payloads),
                log=invocation.request.stage,
            )

    process_generation_task(workspace, first, ShortCallback())
    with session_scope(workspace) as db:
        failed = db.get(GenerationTask, first.id)
        assert failed is not None
        assert failed.state == "failed"
        assert (failed.error or "").startswith("[prose]")
        assert "got 128 body lexical tokens" in (failed.error or "")
        retry_generation_task(db, first.id)
        retry = claim_generation_task(db)
        assert retry is not None

    observed: list[str] = []

    class ObserveRetry:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            observed.append(invocation.request.stage)
            return CallbackResult(exit_code=1, payload="", log="stop after observing")

    process_generation_task(workspace, retry, ObserveRetry())

    assert observed == ["prose"]


def test_retry_reuses_every_valid_stage_and_runs_only_failed_grammar(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        first = claim_generation_task(db)
        assert first is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }

    class FailingGrammar:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            if invocation.request.stage == "grammar":
                return CallbackResult(exit_code=1, payload="", log="grammar failed")
            return CallbackResult(
                exit_code=0,
                payload=_staged_payload(invocation, payloads),
                log=invocation.request.stage,
            )

    process_generation_task(workspace, first, FailingGrammar())
    with session_scope(workspace) as db:
        failed = db.get(GenerationTask, first.id)
        assert failed is not None and failed.state == "failed"
        retry_generation_task(db, first.id)
        retry = claim_generation_task(db)
        assert retry is not None and retry.attempts == 2

    observed: list[str] = []

    class SuccessfulRetry:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            observed.append(invocation.request.stage)
            assert invocation.request.stage == "grammar"
            assert invocation.request.tokenized_lessons
            return CallbackResult(exit_code=0, payload=grammar.model_dump_json(), log="ok")

    process_generation_task(workspace, retry, SuccessfulRetry())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, retry.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
    assert observed == ["grammar"]


def test_later_stage_gets_its_own_repair_attempt_and_reuses_dependencies(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        first = claim_generation_task(db)
        assert first is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }

    class LexicalFailure:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            if invocation.request.stage == "lexical":
                return CallbackResult(exit_code=1, payload="", log="lexical failed")
            return CallbackResult(
                exit_code=0,
                payload=_staged_payload(invocation, payloads),
                log=invocation.request.stage,
            )

    process_generation_task(workspace, first, LexicalFailure())
    with session_scope(workspace) as db:
        retry_generation_task(db, first.id)
        second = claim_generation_task(db)
        assert second is not None and second.attempts == 2

    second_stages: list[str] = []

    class GrammarFailure:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            second_stages.append(invocation.request.stage)
            if invocation.request.stage == "grammar":
                return CallbackResult(exit_code=1, payload="", log="grammar failed")
            return CallbackResult(
                exit_code=0,
                payload=_staged_payload(invocation, payloads),
                log=invocation.request.stage,
            )

    process_generation_task(workspace, second, GrammarFailure())
    assert second_stages.count("lexical") == 1  # one batched call covers all four sentences
    assert second_stages[-1] == "grammar"
    with session_scope(workspace) as db:
        failed = db.get(GenerationTask, second.id)
        assert failed is not None
        assert failed.state == "failed"
        assert failed.payload["generation_stage_failures"] == {
            "grammar": 1,
            "lexical": 1,
        }
        retry_generation_task(db, second.id)
        third = claim_generation_task(db)
        assert third is not None and third.attempts == 3

    third_stages: list[str] = []

    class GrammarRepair:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            third_stages.append(invocation.request.stage)
            return CallbackResult(exit_code=0, payload=grammar.model_dump_json(), log="grammar")

    process_generation_task(workspace, third, GrammarRepair())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, third.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
        assert db.scalar(select(func.count()).select_from(Lesson)) == 1
    assert third_stages == ["grammar"]


def test_repeated_same_stage_failure_exhausts_only_that_repair_budget(
    db: Session,
    tmp_path: Path,
) -> None:
    update_profile(db, {"level": "A1"})
    task = ensure_generation_task(db, require_enabled=False)
    assert task is not None

    first = claim_generation_task(db)
    assert first is not None
    fail_generation_task(
        db,
        first.id,
        log_path=str(tmp_path / "grammar-1.log"),
        error="[grammar] invalid callback response: bad range",
    )
    maintain_generation_task(db)

    second = claim_generation_task(db)
    assert second is not None and second.attempts == 2
    failed = fail_generation_task(
        db,
        second.id,
        log_path=str(tmp_path / "grammar-2.log"),
        error="[grammar] invalid callback response: bad range again",
    )

    assert failed.payload["generation_stage_failures"] == {"grammar": 2}
    assert maintain_generation_task(db) is failed
    assert claim_generation_task(db) is None
    with pytest.raises(ValueError, match="maximum attempt count"):
        retry_generation_task(db, failed.id)


def test_retry_reuses_lexical_and_grammar_when_only_translation_failed(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        first = claim_generation_task(db)
        assert first is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }

    class FailingTranslation:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            if invocation.request.stage == "translation":
                return CallbackResult(exit_code=1, payload="", log="translation failed")
            return CallbackResult(
                exit_code=0,
                payload=_staged_payload(invocation, payloads),
                log=invocation.request.stage,
            )

    process_generation_task(workspace, first, FailingTranslation())
    with session_scope(workspace) as db:
        failed = db.get(GenerationTask, first.id)
        assert failed is not None and failed.state == "failed"
        retry_generation_task(db, first.id)
        retry = claim_generation_task(db)
        assert retry is not None

    observed: list[str] = []

    class TranslationRetry:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            observed.append(invocation.request.stage)
            return CallbackResult(
                exit_code=0,
                payload=translation.model_dump_json(),
                log="translation",
            )

    process_generation_task(workspace, retry, TranslationRetry())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, retry.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
    assert observed == ["translation"]


def test_assembly_retry_reuses_all_valid_callback_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server import agent_worker

    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        first = claim_generation_task(db)
        assert first is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
        "grammar": grammar.model_dump_json(),
    }

    class CompleteCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            return CallbackResult(
                exit_code=0,
                payload=_staged_payload(invocation, payloads),
                log=invocation.request.stage,
            )

    original_import = agent_worker._import_callback_lessons
    import_calls = 0

    def flaky_import(*args: Any, **kwargs: Any) -> None:
        nonlocal import_calls
        import_calls += 1
        if import_calls == 1:
            raise ValueError("synthetic assembly interruption")
        original_import(*args, **kwargs)

    monkeypatch.setattr(agent_worker, "_import_callback_lessons", flaky_import)
    process_generation_task(workspace, first, CompleteCallback())
    with session_scope(workspace) as db:
        failed = db.get(GenerationTask, first.id)
        assert failed is not None
        assert failed.state == "failed"
        assert (failed.error or "").startswith("[assembly]")
        retry_generation_task(db, first.id)
        retry = claim_generation_task(db)
        assert retry is not None

    class NoCallbackExpected:
        def run(self, _invocation: CallbackInvocation) -> CallbackResult:
            raise AssertionError("validated stages must be reused after an assembly failure")

    process_generation_task(workspace, retry, NoCallbackExpected())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, retry.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
    assert import_calls == 2


def test_host_sorts_grammar_ranges_and_assigns_deterministic_keys(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
    }

    class GrammarCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            if invocation.request.stage != "grammar":
                return CallbackResult(
                    exit_code=0,
                    payload=_staged_payload(invocation, payloads),
                    log=invocation.request.stage,
                )
            construction_key = invocation.request.grammar_catalog[0].key
            wire = grammar.model_dump(mode="json")
            wire["lessons"][0]["sentences"][1]["occurrences"] = [
                {
                    "construction_key": construction_key,
                    "run_start": 2,
                    "run_end": 3,
                    "note": "Later range.",
                },
                {
                    "construction_key": construction_key,
                    "run_start": 0,
                    "run_end": 1,
                    "note": "Earlier range.",
                },
            ]
            enriched = GenerationGrammarResult.model_validate(wire)
            return CallbackResult(
                exit_code=0,
                payload=enriched.model_dump_json(),
                log="grammar",
            )

    process_generation_task(workspace, claimed, GrammarCallback())

    with session_scope(workspace) as db:
        lesson = db.scalar(select(Lesson))
        assert lesson is not None
        document = LessonDocument.model_validate(lesson.payload)
    occurrences = document.blocks[0].sentences[0].grammar
    assert [occurrence.key for occurrence in occurrences] == ["grammar:2:1", "grammar:2:2"]
    assert [occurrence.run_start for occurrence in occurrences] == [0, 2]


def test_host_drops_unknown_grammar_and_clamps_one_past_sentence_end(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
    }
    allowed_key: str | None = None

    class GrammarCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            nonlocal allowed_key
            if invocation.request.stage != "grammar":
                return CallbackResult(
                    exit_code=0,
                    payload=_staged_payload(invocation, payloads),
                    log=invocation.request.stage,
                )
            allowed_key = invocation.request.grammar_catalog[0].key
            run_count = len(invocation.request.tokenized_lessons[0].blocks[0].sentences[0].runs)
            wire = grammar.model_dump(mode="json")
            wire["lessons"][0]["sentences"][1]["occurrences"] = [
                {
                    "construction_key": "es:invented-construction",
                    "run_start": 0,
                    "run_end": 1,
                    "note": None,
                },
                {
                    "construction_key": allowed_key,
                    "run_start": 0,
                    "run_end": run_count + 1,
                    "note": "Inclusive end by mistake.",
                },
            ]
            enriched = GenerationGrammarResult.model_validate(wire)
            return CallbackResult(
                exit_code=0,
                payload=enriched.model_dump_json(),
                log="grammar",
            )

    process_generation_task(workspace, claimed, GrammarCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
        lesson = db.scalar(select(Lesson))
        assert lesson is not None
        document = LessonDocument.model_validate(lesson.payload)
    sentence = document.blocks[0].sentences[0]
    assert allowed_key is not None
    assert [occurrence.construction_key for occurrence in sentence.grammar] == [allowed_key]
    assert sentence.grammar[0].run_end == len(sentence.runs)


def test_host_drops_unrecoverable_grammar_range_without_failing_lesson(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    prose, lexical, translation, grammar, _response = _staged_callback_fixture()
    payloads = {
        "prose": prose.model_dump_json(),
        "lexical": lexical.model_dump_json(),
        "translation": translation.model_dump_json(),
    }
    allowed_key: str | None = None

    class GrammarCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            nonlocal allowed_key
            if invocation.request.stage != "grammar":
                return CallbackResult(
                    exit_code=0,
                    payload=_staged_payload(invocation, payloads),
                    log=invocation.request.stage,
                )
            allowed_key = invocation.request.grammar_catalog[0].key
            run_count = len(invocation.request.tokenized_lessons[0].blocks[0].sentences[0].runs)
            wire = grammar.model_dump(mode="json")
            wire["lessons"][0]["sentences"][1]["occurrences"] = [
                {
                    "construction_key": allowed_key,
                    "run_start": 0,
                    "run_end": 1,
                    "note": "Valid sibling.",
                },
                {
                    "construction_key": allowed_key,
                    "run_start": run_count + 1,
                    "run_end": run_count + 3,
                    "note": "Unrecoverable range.",
                },
            ]
            enriched = GenerationGrammarResult.model_validate(wire)
            return CallbackResult(
                exit_code=0,
                payload=enriched.model_dump_json(),
                log="grammar",
            )

    process_generation_task(workspace, claimed, GrammarCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
        lesson = db.scalar(select(Lesson))
        assert lesson is not None
        document = LessonDocument.model_validate(lesson.payload)
    sentence = document.blocks[0].sentences[0]
    assert allowed_key is not None
    assert [occurrence.construction_key for occurrence in sentence.grammar] == [allowed_key]
    assert sentence.grammar[0].note == "Valid sibling."


def _chinese_generated_draft(runs: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _generated_draft(runs)
    payload["title"] = "测试"
    payload["title_sentence"] = {
        "key": "title-sentence",
        "runs": [_term_run("测试", "zh:test:NOUN")],
        "translation": "Test",
    }
    return payload


def test_generation_instructions_append_profile_language_guidance(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).create(
        "zh-hans",
        learning_language="zh-Hans",
        translation_language="en",
    )
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1"})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    _, invocation = _prepare_invocation(workspace, claimed)
    try:
        assert "For Chinese" not in LEXICAL_COVERAGE_INSTRUCTIONS
        assert "一个 -> 一 + 个" in invocation.request.instructions
        assert "Keep lexicalized entries intact" in invocation.request.instructions
        assert "annotate every clearly present cataloged construction" not in (
            invocation.request.instructions
        )
        assert invocation.base_request is not None
        assert "annotate every clearly present cataloged construction" not in (
            invocation.base_request.instructions
        )
        assert "priority_grammar is a ranked choice pool, not a checklist" in (
            invocation.request.instructions
        )
        assert invocation.base_request.instructions == invocation.request.instructions
        assert "zero realized choices is valid" in invocation.request.instructions
        assert "Content variety is a required quality constraint" in invocation.request.instructions
        assert "generic cooperative success story" in invocation.request.instructions
        assert TERM_IDENTITY_PREFLIGHT_INSTRUCTIONS not in invocation.request.instructions
        assert TERM_IDENTITY_PREFLIGHT_INSTRUCTIONS not in invocation.base_request.instructions
        assert "lesson-level terms catalog" not in invocation.base_request.instructions
        assert invocation.request.content_plan.archetype
        assert invocation.request.content_plan.concrete_seed
    finally:
        invocation.job_dir.rmdir()


def test_content_plan_avoids_recent_structure_and_names_recent_situations(
    db: Session, lesson_factory: Any
) -> None:
    first = build_content_plan(
        build_agent_brief(db),
        task_id=4,
        requested_topic=None,
    )
    import_lesson(
        db,
        lesson_factory(
            metadata={"content_plan": first.model_dump(mode="json")},
        ),
    )

    next_plan = build_content_plan(
        build_agent_brief(db),
        task_id=4,
        requested_topic=None,
    )

    assert next_plan.archetype != first.archetype
    assert next_plan == build_content_plan(
        build_agent_brief(db),
        task_id=4,
        requested_topic=None,
    )
    # Recent subjects reach the agent through the brief's 20-text content history.
    assert [item.title for item in build_agent_brief(db).content_history] == ["Lesson lesson-one"]
    assert any("content_history" in warning for warning in next_plan.avoid_patterns)
    assert any("group gathers" in warning for warning in next_plan.avoid_patterns)


def test_grammar_opportunity_budget_adapts_without_becoming_a_checklist(db: Session) -> None:
    brief = build_agent_brief(db)
    assert len(brief.priority_grammar) >= 4

    uncertain = brief.model_copy(
        update={
            "profile": brief.profile.model_copy(
                update={
                    "difficulty": 0.1,
                    "proficiency": brief.profile.proficiency.model_copy(
                        update={"source": "unknown", "lower": None, "upper": None}
                    ),
                }
            ),
        }
    )
    confident = brief.model_copy(
        update={
            "profile": brief.profile.model_copy(
                update={
                    "difficulty": 0.9,
                    "proficiency": brief.profile.proficiency.model_copy(
                        update={"source": "estimated", "lower": 0.48, "upper": 0.52}
                    ),
                }
            ),
        }
    )

    short_less = _grammar_generation_policy(
        uncertain,
        target_text_length=120,
        calibration=False,
        feedback=("less_grammar",),
    )
    long_more = _grammar_generation_policy(
        confident,
        target_text_length=720,
        calibration=False,
        feedback=("more_grammar",),
    )
    ordinary_less = _grammar_generation_policy(
        brief,
        target_text_length=420,
        calibration=False,
        feedback=("less_grammar",),
    )
    ordinary_more = _grammar_generation_policy(
        brief,
        target_text_length=420,
        calibration=False,
        feedback=("more_grammar",),
    )
    calibration = _grammar_generation_policy(
        confident,
        target_text_length=720,
        calibration=True,
        feedback=("more_grammar",),
    )

    assert short_less.preferred_count <= short_less.max_count < long_more.max_count
    assert ordinary_less.preferred_count < ordinary_more.preferred_count
    assert long_more.max_count <= 4
    assert len(ordinary_more.offered_keys) > ordinary_more.max_count
    assert calibration.preferred_count == calibration.max_count == 0
    assert calibration.offered_keys == ()

    instructions = _grammar_generation_instructions(
        brief,
        ordinary_more,
        calibration=False,
    )
    assert "smallest accurate half-open run_start/run_end range" in instructions
    assert "Inspect beyond the first item" in instructions
    assert "Natural comprehensible text quality wins" in instructions


def test_ordinary_generation_quality_contract_is_explicit_and_calibration_is_exempt() -> None:
    instructions = _generation_quality_instructions(
        target_difficulty=0.3,
        target_text_length=300,
    )
    assert "requested numeric lesson difficulty is 0.300" in instructions
    assert "description of the resulting prose, not a label" in instructions
    assert "within 0.05 of the requested value" in instructions
    assert "full 300 body lexical tokens" in instructions
    assert "rejects fewer than 180" in instructions

    acceptable = GeneratedLessonDraft.model_validate(
        _callback_sized_generated_draft(
            [_term_run("hola", "es:hola:WORD")],
            target_length=30,
            difficulty=0.349,
        )
    )
    _validate_generated_lesson_quality(
        acceptable,
        mode="lesson",
        target_difficulty=0.3,
        target_text_length=50,
    )

    too_short = GeneratedLessonDraft.model_validate(
        _callback_sized_generated_draft(
            [_term_run("hola", "es:hola:WORD")],
            target_length=29,
            difficulty=0.3,
        )
    )
    with pytest.raises(ValueError, match=r"got 29 body lexical tokens.*minimum 30"):
        _validate_generated_lesson_quality(
            too_short,
            mode="lesson",
            target_difficulty=0.3,
            target_text_length=50,
        )

    wrong_difficulty = acceptable.model_copy(update={"difficulty": 0.351})
    with pytest.raises(ValueError, match=r"got 0.351, requested 0.300"):
        _validate_generated_lesson_quality(
            wrong_difficulty,
            mode="lesson",
            target_difficulty=0.3,
            target_text_length=50,
        )

    compact_calibration = GeneratedLessonDraft.model_validate(
        _generated_draft([_term_run("probe", "es:probe:WORD")])
    )
    _validate_generated_lesson_quality(
        compact_calibration,
        mode="calibration",
        target_difficulty=0.9,
        target_text_length=300,
    )


def test_worker_imports_generated_units_with_known_host_decomposition(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).create(
        "zh-hans",
        learning_language="zh-Hans",
        translation_language="en",
    )
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1"})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    composite = _term_run("一个", "zh:yige:QUANTIFIER", gloss="one; a")
    composite["term"]["pos"] = "quantifier"
    payload = _chinese_generated_draft([])
    payload["blocks"][0]["sentences"] = [
        {
            "key": f"sentence-{index}",
            "runs": [deepcopy(composite) for _ in range(100)],
            "translation": "Quality-contract fixture.",
        }
        for index in range(1, 4)
    ]
    response = GenerationCallbackResult.model_validate(
        {
            "schema_version": 1,
            "lessons": [payload],
        }
    )

    class FixedCallback:
        def run(self, _invocation: CallbackInvocation) -> CallbackResult:
            return CallbackResult(exit_code=0, payload=response.model_dump_json(), log="ok")

    process_generation_task(workspace, claimed, FixedCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
        assert stored.error is None
        assert db.scalar(select(func.count()).select_from(Lesson)) == 1


def test_worker_repairs_partial_surface_with_exact_stored_component(
    tmp_path: Path,
    lesson_factory: Any,
) -> None:
    workspace = WorkspaceRegistry(tmp_path).create(
        "zh-hans",
        learning_language="zh-Hans",
        translation_language="en",
    )
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        import_lesson(
            db,
            lesson_factory(
                key="known-kind-classifier",
                learning_language="zh-Hans",
                terms=[("zh:zhong:CL", "种", "种", "classifier", "kind; type", 450)],
                targets=[],
            ),
        )
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    partial = _term_run("这种", "zh:zhe:PRON", gloss="this")
    partial["term"].update(
        {
            "lemma": "这",
            "pos": "pronoun",
            "pronunciation": "zhè",
            "frequency_rank": 20,
        }
    )
    payload = _chinese_generated_draft([])
    payload["blocks"][0]["sentences"] = [
        {
            "key": f"sentence-{index}",
            "runs": [deepcopy(partial) for _ in range(50)],
            "translation": "Quality-contract fixture.",
        }
        for index in range(1, 7)
    ]
    response = GenerationCallbackResult.model_validate({"schema_version": 1, "lessons": [payload]})

    class FixedCallback:
        def run(self, _invocation: CallbackInvocation) -> CallbackResult:
            return CallbackResult(exit_code=0, payload=response.model_dump_json(), log="ok")

    process_generation_task(workspace, claimed, FixedCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed", stored.error
        generated = db.scalars(select(Lesson).order_by(Lesson.id.desc())).first()
        assert generated is not None
        document = LessonDocument.model_validate(generated.payload)
        runs = document.blocks[0].sentences[0].runs[:2]
        assert [run.text for run in runs] == ["这", "种"]
        assert [run.term.key if run.term is not None else None for run in runs] == [
            "zh:zhe:PRON",
            "zh:zhong:CL",
        ]


def test_worker_rejects_an_ordinary_callback_that_is_drastically_too_short(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A2", "difficulty": 0.3})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    response = GenerationCallbackResult.model_validate(
        {
            "schema_version": 1,
            "lessons": [
                _callback_sized_generated_draft(
                    [_term_run("hola", "es:hola:WORD")],
                    target_length=128,
                    difficulty=0.3,
                )
            ],
        }
    )

    class ShortCallback:
        def run(self, _invocation: CallbackInvocation) -> CallbackResult:
            return CallbackResult(exit_code=0, payload=response.model_dump_json(), log="ok")

    process_generation_task(workspace, claimed, ShortCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "failed"
        assert "got 128 body lexical tokens, requested 300, minimum 180" in (stored.error or "")
        assert db.scalar(select(func.count()).select_from(Lesson)) == 0


def test_worker_imports_one_lesson_then_maintenance_queues_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "3")
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1"})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    requested_counts: list[int] = []

    class FixedCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            requested_counts.append(invocation.request.lesson_count)
            payload = _callback_sized_generated_draft(
                [_term_run("hola", "es:hola:WORD")],
                difficulty=invocation.request.brief.profile.difficulty,
            )
            response = GenerationCallbackResult.model_validate(
                {"schema_version": 1, "lessons": [payload]}
            )
            return CallbackResult(exit_code=0, payload=response.model_dump_json(), log="ok")

    process_generation_task(workspace, claimed, FixedCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed"
        assert db.scalar(select(func.count()).select_from(Lesson)) == 1
    assert requested_counts == [1]

    _maintain_worker_workspaces([workspace])

    with session_scope(workspace) as db:
        tasks = db.scalars(select(GenerationTask).order_by(GenerationTask.id)).all()
        assert [task.state for task in tasks] == ["completed", "pending"]
        assert tasks[-1].payload["needed_lesson_count"] == 1
        assert tasks[-1].payload["queue_shortfall"] == 2


def test_generated_lesson_metadata_records_offered_and_realized_grammar(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1", "difficulty": 0.1})
        task = ensure_generation_task(db, require_enabled=False)
        assert task is not None
        claimed = claim_generation_task(db)
        assert claimed is not None

    chosen_keys: list[str] = []
    requested_difficulties: list[float] = []
    requested_levels: list[str] = []
    generated_difficulties: list[float] = []

    class GrammarCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            chosen = invocation.request.brief.priority_grammar[0].key
            chosen_keys.append(chosen)
            target = invocation.request.brief.profile.difficulty
            declared = target + 0.04 if target <= 0.96 else target - 0.04
            requested_difficulties.append(target)
            requested_levels.append(invocation.request.brief.profile.level)
            generated_difficulties.append(declared)
            payload = _callback_sized_generated_draft(
                [_term_run("habla", "es:habla:WORD")],
                difficulty=declared,
            )
            payload["level"] = "C2"
            payload["blocks"][0]["sentences"][0]["grammar"] = [
                {
                    "key": "sentence-1:grammar:1",
                    "construction_key": chosen,
                    "run_start": 0,
                    "run_end": 1,
                    "note": "Context-specific use.",
                }
            ]
            response = GenerationCallbackResult.model_validate(
                {"schema_version": 1, "lessons": [payload]}
            )
            return CallbackResult(exit_code=0, payload=response.model_dump_json(), log="ok")

    process_generation_task(workspace, claimed, GrammarCallback())

    with session_scope(workspace) as db:
        generated = db.scalar(select(Lesson))
        assert generated is not None
        assert generated.level == requested_levels[0]
        assert generated.difficulty == pytest.approx(requested_difficulties[0])
        assert generated.metadata_json["adaptive_target_difficulty"] == pytest.approx(
            requested_difficulties[0]
        )
        assert generated.metadata_json["generated_difficulty"] == pytest.approx(
            generated_difficulties[0]
        )
        grammar = generated.metadata_json["grammar"]
        assert chosen_keys == [grammar["offered"][0]]
        assert grammar["realized"] == chosen_keys
        assert grammar["annotated"] == chosen_keys
        assert grammar["allow_zero"] is True
        assert 0 <= grammar["preferred"] <= grammar["max"] <= 4
        content_plan = generated.metadata_json["content_plan"]
        assert content_plan["archetype"]
        assert content_plan["concrete_seed"]


def test_topic_request_coexists_with_a_running_queue_fill(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "3")
    update_profile(db, {"level": "A1"})
    queue_task = ensure_generation_task(db, require_enabled=False)
    assert queue_task is not None
    running = claim_generation_task(db)
    assert running is not None
    assert running.id == queue_task.id

    request_id = uuid4()
    requested = request_topic_lesson(
        db,
        TextRequestIn(request_id=request_id, topic="marine archaeology"),
    )

    assert running.state == "running"
    assert requested.state == "pending"
    assert requested.id != running.id
    assert requested.dedupe_key == f"topic-request:{request_id}"
    assert requested.payload["request_kind"] == "topic_request"
    assert requested.payload["requested_topic"] == "marine archaeology"


def test_running_topic_request_recovers_with_its_durable_identity(db: Session) -> None:
    request_id = uuid4()
    topic = request_topic_lesson(
        db,
        TextRequestIn(request_id=request_id, topic="volcanic islands"),
    )
    running = claim_generation_task(db)
    assert running is not None
    assert running.id == topic.id

    assert recover_running_generation_tasks(db) == 1

    recovered = db.get(GenerationTask, topic.id)
    assert recovered is not None
    assert recovered.state == "pending"
    assert recovered.dedupe_key == f"topic-request:{request_id}"
    assert recovered.payload["requested_topic"] == "volcanic islands"
    assert recovered.error == "worker stopped before reporting a result"
    assert recovered.payload["previous_failures"] == [
        {
            "task_id": recovered.id,
            "attempt": 1,
            "error": "worker stopped before reporting a result",
        }
    ]


def test_second_worker_crash_is_terminal_and_cannot_claim_a_third_attempt(db: Session) -> None:
    request_id = uuid4()
    topic = request_topic_lesson(
        db,
        TextRequestIn(request_id=request_id, topic="volcanic islands"),
    )
    first = claim_generation_task(db)
    assert first is not None and first.id == topic.id and first.attempts == 1
    assert recover_running_generation_tasks(db) == 1

    second = claim_generation_task(db)
    assert second is not None and second.id == topic.id and second.attempts == 2
    assert recover_running_generation_tasks(db) == 1

    terminal = db.get(GenerationTask, topic.id)
    assert terminal is not None
    assert terminal.state == "failed"
    assert terminal.attempts == 2
    assert terminal.error == "worker stopped before reporting a result"
    assert [failure["attempt"] for failure in terminal.payload["previous_failures"]] == [1, 2]
    assert claim_generation_task(db) is None
    with pytest.raises(ValueError, match="maximum attempt count"):
        retry_generation_task(db, topic.id)


def test_failed_topic_request_retries_without_blocking_queue_maintenance(
    db: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "3")
    update_profile(db, {"level": "A1"})
    topic = request_topic_lesson(
        db,
        TextRequestIn(request_id=uuid4(), topic="railway history"),
    )
    claimed = claim_generation_task(db)
    assert claimed is not None
    assert claimed.id == topic.id
    fail_generation_task(
        db,
        claimed.id,
        log_path=str(tmp_path / "topic-failure.log"),
        error="temporary failure",
    )

    queue = ensure_generation_task(db, require_enabled=False)
    retried = maintain_topic_generation_tasks(db)

    assert queue is not None
    assert queue.id != topic.id
    assert queue.payload["request_kind"] == "queue_fill"
    assert [task.id for task in retried] == [topic.id]
    assert retried[0].state == "pending"
    assert db.scalars(
        select(GenerationTask).where(GenerationTask.state == "pending").order_by(GenerationTask.id)
    ).all() == [retried[0], queue]


def test_topic_request_imports_one_extra_lesson_when_queue_is_full(
    tmp_path: Path,
    lesson_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARC_LANG_QUEUE_TARGET", "1")
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    request_id = uuid4()
    with session_scope(workspace) as db:
        _activate_test_profile(db)
        update_profile(db, {"level": "A1"})
        import_lesson(db, lesson_factory(key="already-queued"))
        assert ensure_generation_task(db, require_enabled=False) is None
        task = request_topic_lesson(
            db,
            TextRequestIn(request_id=request_id, topic="deep-sea exploration"),
        )
        # Explicit requests remain valid while queued and use the freshest profile at claim time.
        update_profile(db, {"difficulty": 0.2})
        claimed = claim_generation_task(db)
        assert claimed is not None
        assert claimed.id == task.id

    response = GenerationCallbackResult.model_validate(
        {
            "schema_version": 1,
            "lessons": [
                _callback_sized_generated_draft(
                    [_term_run("hola", "es:hola:WORD")],
                    difficulty=0.2,
                )
            ],
        }
    )
    observed: list[GenerationCallbackRequest] = []

    class FixedCallback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            observed.append(invocation.request)
            return CallbackResult(exit_code=0, payload=response.model_dump_json(), log="ok")

    process_generation_task(workspace, claimed, FixedCallback())

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, claimed.id)
        assert stored is not None
        assert stored.state == "completed"
        assert stored.dedupe_key == f"topic-request:{request_id}"
        lessons = db.scalars(select(Lesson).order_by(Lesson.id)).all()
        assert len(lessons) == 2
        generated = lessons[-1]
        assert generated.topic == "deep-sea exploration"
        assert generated.metadata_json["request_id"] == str(request_id)
        assert generated.metadata_json["requested_topic"] == "deep-sea exploration"
        assert generated.metadata_json["generated_topic"] == "testing"
        replay = request_topic_lesson(
            db,
            TextRequestIn(request_id=request_id, topic="deep-sea exploration"),
        )
        assert replay.id == stored.id

    assert len(observed) == 1
    assert observed[0].lesson_count == 1
    assert observed[0].request_kind == "topic_request"
    assert observed[0].requested_topic == "deep-sea exploration"
    assert observed[0].brief.profile.difficulty == 0.2
    assert "explicit topic overrides" in observed[0].instructions
    assert "deep-sea exploration" in observed[0].content_plan.concrete_seed


@pytest.mark.parametrize("plain_text", ["hola mundo", "今天天气很好"])
def test_generated_drafts_reject_sparse_lexical_coverage(plain_text: str) -> None:
    payload = _generated_draft([{"text": plain_text}, {"text": "。"}])

    with pytest.raises(ValidationError, match="annotate every lexical token"):
        GeneratedLessonDraft.model_validate(payload)

    manual = LessonDocument(
        key=f"manual-sparse-{len(plain_text)}",
        title="Backward-compatible manual lesson",
        learning_language="es-ES",
        translation_language="en",
        blocks=payload["blocks"],
    )
    assert manual.blocks[0].sentences[0].text == f"{plain_text}。"


def test_generated_drafts_require_a_fully_annotated_matching_title() -> None:
    payload = _generated_draft([_term_run("hola", "es:hola:WORD")])
    del payload["title_sentence"]
    with pytest.raises(ValidationError, match="title_sentence"):
        GeneratedLessonDraft.model_validate(payload)

    mismatched = _generated_draft([_term_run("hola", "es:hola:WORD")])
    mismatched["title_sentence"]["runs"][0]["text"] = "Otra"
    with pytest.raises(ValidationError, match="exactly equal title"):
        GeneratedLessonDraft.model_validate(mismatched)

    unannotated = _generated_draft([_term_run("hola", "es:hola:WORD")])
    unannotated["title_sentence"]["runs"] = [{"text": "Prueba"}]
    with pytest.raises(ValidationError, match="annotate every lexical token"):
        GeneratedLessonDraft.model_validate(unannotated)

    untranslated = _generated_draft([_term_run("hola", "es:hola:WORD")])
    untranslated["title_sentence"]["translation"] = None
    with pytest.raises(ValidationError, match="every generated sentence"):
        GeneratedLessonDraft.model_validate(untranslated)


def test_generated_blocks_are_capped_without_limiting_manual_lessons() -> None:
    payload = _generated_draft([_term_run("hola", "es:hola:WORD")])
    sentence = payload["blocks"][0]["sentences"][0]
    payload["blocks"][0]["sentences"] = [
        {**deepcopy(sentence), "key": f"sentence-{index}"} for index in range(1, 23)
    ]

    with pytest.raises(ValidationError) as caught:
        GeneratedLessonDraft.model_validate(payload)

    assert caught.value.errors()[0]["loc"] == ("blocks", 0, "sentences")
    assert caught.value.errors()[0]["type"] == "too_long"

    manual = LessonDocument(
        key="manual-long-block",
        title="Manual long block",
        learning_language="es-ES",
        translation_language="en",
        blocks=payload["blocks"],
    )
    assert len(manual.blocks[0].sentences) == 22


def test_manual_annotated_title_joins_catalog_and_requires_a_unique_key() -> None:
    payload = _generated_draft([_term_run("hola", "es:hola:WORD")])
    manual = LessonDocument(
        key="manual-annotated-title",
        title=payload["title"],
        title_sentence=payload["title_sentence"],
        learning_language="es-ES",
        translation_language="en",
        blocks=payload["blocks"],
    )

    assert "es:prueba:NOUN" in manual.term_catalog()
    assert manual.sentence_terms()["title-sentence"] == {"es:prueba:NOUN"}

    duplicate_key = deepcopy(payload["title_sentence"])
    duplicate_key["key"] = "sentence-1"
    with pytest.raises(ValidationError, match="duplicate sentence key"):
        LessonDocument(
            key="manual-duplicate-title-key",
            title=payload["title"],
            title_sentence=duplicate_key,
            learning_language="es-ES",
            translation_language="en",
            blocks=payload["blocks"],
        )


def test_generated_chinese_draft_accepts_word_segmentation() -> None:
    draft = GeneratedLessonDraft.model_validate(
        _generated_draft(
            [
                _term_run("今天", "zh:今天:NOUN"),
                _term_run("天气", "zh:天气:NOUN"),
                _term_run("很", "zh:很:ADV"),
                _term_run("好", "zh:好:ADJ"),
                {"text": "。"},
            ],
            targets=["zh:天气:NOUN"],
        )
    )

    terms = [
        run.term
        for block in draft.blocks
        for sentence in block.sentences
        for run in sentence.runs
        if run.term is not None
    ]
    assert len(terms) == 4
    assert draft.target_term_keys == ["zh:天气:NOUN"]


def test_callback_response_compacts_terms_and_expands_for_internal_validation() -> None:
    repeated = _term_run("hola", "es:hola:WORD")
    payload = _generated_draft([repeated, {"text": " "}, deepcopy(repeated)])

    response = GenerationCallbackResult.model_validate({"schema_version": 1, "lessons": [payload]})

    wire = response.model_dump(mode="json")
    lesson = wire["lessons"][0]
    assert [term["key"] for term in lesson["terms"]] == [
        "es:prueba:NOUN",
        "es:hola:WORD",
    ]
    body_runs = lesson["blocks"][0]["sentences"][0]["runs"]
    assert [run["term_key"] for run in body_runs] == [
        "es:hola:WORD",
        None,
        "es:hola:WORD",
    ]
    assert all("term" not in run for run in body_runs)

    expanded = response.expanded_lessons()[0]
    first = expanded.blocks[0].sentences[0].runs[0].term
    second = expanded.blocks[0].sentences[0].runs[2].term
    assert first == second
    assert first is not None
    assert first.gloss == "gloss for hola"


def test_callback_response_defers_unknown_term_references_but_locates_expand_error() -> None:
    response = GenerationCallbackResult.model_validate(
        {
            "schema_version": 1,
            "lessons": [_generated_draft([_term_run("hola", "es:hola:WORD")])],
        }
    )
    unknown = response.model_dump(mode="json")
    unknown["lessons"][0]["blocks"][0]["sentences"][0]["runs"][0]["term_key"] = "missing"
    parsed = GenerationCallbackResult.model_validate(unknown)
    with pytest.raises(
        ValueError,
        match=r"unknown term key 'missing'.*sentence 'sentence-1', run 0, surface 'hola'",
    ):
        parsed.expanded_lessons()

    duplicate = response.model_dump(mode="json")
    duplicate["lessons"][0]["terms"].append(deepcopy(duplicate["lessons"][0]["terms"][0]))
    with pytest.raises(ValidationError, match="defines key more than once"):
        GenerationCallbackResult.model_validate(duplicate)


def test_callback_catalog_allows_valid_unreferenced_decomposition_components() -> None:
    response = GenerationCallbackResult.model_validate(
        {
            "schema_version": 1,
            "lessons": [_generated_draft([_term_run("笑了", "zh:xiaole:WORD")])],
        }
    )
    wire = response.model_dump(mode="json")
    wire["lessons"][0]["terms"].append(
        {
            "key": "zh:xiao:VERB",
            "lemma": "笑",
            "pos": "verb",
            "gloss": "to laugh",
            "pronunciation": "xiào",
            "frequency_rank": 900,
        }
    )

    reparable = GenerationCallbackResult.model_validate(wire)

    assert reparable.lessons[0].term_catalog()["zh:xiao:VERB"].lemma == "笑"


def test_callback_normalization_splits_exact_catalog_backed_chinese_components() -> None:
    composite = _term_run("一个", "zh:yige:QUANTIFIER", gloss="one item")
    composite["term"]["pos"] = "quantifier"
    payload = _chinese_generated_draft([composite])
    payload["target_term_keys"] = ["zh:yige:QUANTIFIER"]
    payload["blocks"][0]["sentences"][0]["grammar"] = [
        {
            "key": "sentence-1:grammar:1",
            "construction_key": "zh:measure-word-phrase",
            "run_start": 0,
            "run_end": 1,
        }
    ]
    response = GenerationCallbackResult.model_validate({"schema_version": 1, "lessons": [payload]})
    wire = response.model_dump(mode="json")
    wire["lessons"][0]["terms"].extend(
        [
            {
                "key": "zh:yi:NUM",
                "lemma": "一",
                "pos": "numeral",
                "gloss": "one",
                "pronunciation": "yī",
                "frequency_rank": 2,
            },
            {
                "key": "zh:ge:CL",
                "lemma": "个",
                "pos": "classifier",
                "gloss": "general classifier",
                "pronunciation": "gè",
                "frequency_rank": 5,
            },
        ]
    )
    lesson = GenerationCallbackResult.model_validate(wire).lessons[0]

    normalized = normalize_callback_lesson(lesson, "zh-Hans")
    sentence = normalized.blocks[0].sentences[0]

    assert "".join(run.text for run in sentence.runs) == "一个"
    assert [run.term_key for run in sentence.runs] == ["zh:yi:NUM", "zh:ge:CL"]
    assert [run.pronunciation for run in sentence.runs] == ["yí", None]
    assert normalized.term_catalog()["zh:yi:NUM"].pronunciation == "yī"
    assert (sentence.grammar[0].run_start, sentence.grammar[0].run_end) == (0, 2)
    assert normalized.target_term_keys == ["zh:yige:QUANTIFIER"]
    assert generated_unit_errors(normalized.expand(), "zh-Hans") == []


def test_callback_normalization_splits_exact_buyao_and_recomputes_sandhi() -> None:
    composite = _term_run("不要", "zh:buyao:AUX", gloss="must not")
    composite["term"].update({"pos": "auxiliary", "pronunciation": "bù yào"})
    composite["pronunciation"] = "bú yào"
    response = GenerationCallbackResult.model_validate(
        {"schema_version": 1, "lessons": [_chinese_generated_draft([composite])]}
    )
    wire = response.model_dump(mode="json")
    wire["lessons"][0]["terms"].extend(
        [
            {
                "key": "zh:bu:ADV",
                "lemma": "不",
                "pos": "adverb",
                "gloss": "not",
                "pronunciation": "bù",
                "frequency_rank": 3,
            },
            {
                "key": "zh:yao:AUX",
                "lemma": "要",
                "pos": "auxiliary",
                "gloss": "should; must",
                "pronunciation": "yào",
                "frequency_rank": 12,
            },
        ]
    )
    lesson = GenerationCallbackResult.model_validate(wire).lessons[0]

    normalized = normalize_callback_lesson(lesson, "zh-Hans")
    sentence = normalized.blocks[0].sentences[0]

    assert [run.text for run in sentence.runs] == ["不", "要"]
    assert [run.term_key for run in sentence.runs] == ["zh:bu:ADV", "zh:yao:AUX"]
    assert [run.pronunciation for run in sentence.runs] == ["bú", None]
    assert normalized.term_catalog()["zh:bu:ADV"].pronunciation == "bù"
    assert normalized.term_catalog()["zh:yao:AUX"].pronunciation == "yào"
    assert generated_unit_errors(normalized.expand(), "zh-Hans") == []


def test_callback_normalization_corrects_sandhi_and_preserves_polyphonic_reading() -> None:
    def run(
        text: str,
        key: str,
        pos: str,
        pronunciation: str,
        *,
        contextual: str | None = None,
    ) -> dict[str, Any]:
        result = _term_run(text, key)
        result["term"].update({"pos": pos, "pronunciation": pronunciation})
        result["pronunciation"] = contextual
        return result

    payload = _chinese_generated_draft(
        [
            run("不", "zh:bu:ADV", "adverb", "bù"),
            run("看", "zh:kan:VERB", "verb", "kàn"),
            {"text": "，"},
            run("一", "zh:yi:NUM", "numeral", "yī", contextual="yí"),
            run("数", "zh:shu:NOUN", "noun", "shù", contextual="shǔ"),
            run("不", "zh:bu:ADV", "adverb", "bù", contextual="bú"),
            run("好", "zh:hao:ADJ", "adjective", "hǎo"),
        ]
    )
    lesson = GenerationCallbackResult.model_validate(
        {"schema_version": 1, "lessons": [payload]}
    ).lessons[0]

    normalized = normalize_callback_lesson(lesson, "zh-Hans")
    runs = normalized.blocks[0].sentences[0].runs

    assert [item.pronunciation for item in runs] == [
        "bú",
        None,
        None,
        "yì",
        "shǔ",
        None,
        None,
    ]
    assert normalized.term_catalog()["zh:yi:NUM"].pronunciation == "yī"
    assert normalized.term_catalog()["zh:bu:ADV"].pronunciation == "bù"
    assert normalized.term_catalog()["zh:shu:NOUN"].pronunciation == "shù"


def test_callback_normalization_repairs_partial_chinese_surface_from_stored_term() -> None:
    partial = _term_run("这种", "zh:zhe:PRON", gloss="this")
    partial["term"].update(
        {
            "lemma": "这",
            "pos": "pronoun",
            "pronunciation": "zhè",
            "frequency_rank": 20,
        }
    )
    response = GenerationCallbackResult.model_validate(
        {"schema_version": 1, "lessons": [_chinese_generated_draft([partial])]}
    )
    kind = LessonTerm(
        key="zh:zhong:CL",
        lemma="种",
        pos="classifier",
        gloss="kind; type",
        pronunciation="zhǒng",
        frequency_rank=450,
    )

    normalized = normalize_callback_lesson(
        response.lessons[0],
        "zh-Hans",
        fallback_terms=[kind],
    )

    assert [run.term_key for run in normalized.blocks[0].sentences[0].runs] == [
        "zh:zhe:PRON",
        "zh:zhong:CL",
    ]
    assert normalized.term_catalog()["zh:zhong:CL"] == kind
    assert generated_unit_errors(normalized.expand(), "zh-Hans") == []

    ambiguous = kind.model_copy(update={"key": "zh:zhong:NOUN", "pos": "noun", "gloss": "species"})
    unrepaired = normalize_callback_lesson(
        response.lessons[0],
        "zh-Hans",
        fallback_terms=[kind, ambiguous],
    )
    errors = generated_unit_errors(unrepaired.expand(), "zh-Hans")
    assert any("surface text must equal" in error for error in errors)


def test_callback_normalization_leaves_contextual_readings_and_probes_intact() -> None:
    partial = _term_run("这种", "zh:zhe:PRON", gloss="this")
    partial["term"].update({"lemma": "这", "pos": "pronoun", "frequency_rank": 20})
    response = GenerationCallbackResult.model_validate(
        {"schema_version": 1, "lessons": [_chinese_generated_draft([partial])]}
    )
    wire = response.model_dump(mode="json")
    body_run = wire["lessons"][0]["blocks"][0]["sentences"][0]["runs"][0]
    body_run["pronunciation"] = "zhè zhǒng"
    contextual = GenerationCallbackResult.model_validate(wire).lessons[0]
    kind = LessonTerm(
        key="zh:zhong:CL",
        lemma="种",
        pos="classifier",
        gloss="kind; type",
        pronunciation="zhǒng",
        frequency_rank=450,
    )

    contextual_result = normalize_callback_lesson(
        contextual,
        "zh-Hans",
        fallback_terms=[kind],
    )

    assert contextual_result.blocks[0].sentences[0].runs == contextual.blocks[0].sentences[0].runs

    composite = _term_run("一个", "zh:yige:QUANTIFIER", gloss="one item")
    composite["term"]["pos"] = "quantifier"
    filler_runs = [_term_run(f"词{index}", f"zh:probe-{index}") for index in range(1, 8)]
    probe_payload = _chinese_generated_draft([composite, *filler_runs])
    probe_keys = ["zh:yige:QUANTIFIER", *(f"zh:probe-{index}" for index in range(1, 8))]
    probe_payload["calibration"] = {
        "sequence": 1,
        "probes": [
            {"term_key": key, "difficulty": index / 10}
            for index, key in enumerate(probe_keys, start=1)
        ],
    }
    probe_response = GenerationCallbackResult.model_validate(
        {"schema_version": 1, "lessons": [probe_payload]}
    )
    probe_wire = probe_response.model_dump(mode="json")
    probe_wire["lessons"][0]["terms"].extend(
        [
            {
                "key": "zh:yi:NUM",
                "lemma": "一",
                "pos": "numeral",
                "gloss": "one",
                "pronunciation": "yī",
                "frequency_rank": 2,
            },
            {
                "key": "zh:ge:CL",
                "lemma": "个",
                "pos": "classifier",
                "gloss": "general classifier",
                "pronunciation": "gè",
                "frequency_rank": 5,
            },
        ]
    )
    probe_lesson = GenerationCallbackResult.model_validate(probe_wire).lessons[0]

    probe_result = normalize_callback_lesson(probe_lesson, "zh-Hans")

    assert probe_result.blocks[0].sentences[0].runs[0].term_key == "zh:yige:QUANTIFIER"


def test_callback_normalization_resolves_exact_missing_stored_term_key() -> None:
    response = GenerationCallbackResult.model_validate(
        {
            "schema_version": 1,
            "lessons": [_chinese_generated_draft([_term_run("很", "hen-very")])],
        }
    )
    wire = response.model_dump(mode="json")
    wire["lessons"][0]["terms"] = [
        term for term in wire["lessons"][0]["terms"] if term["key"] != "hen-very"
    ]
    unresolved = GenerationCallbackResult.model_validate(wire).lessons[0]
    known = LessonTerm(
        key="hen-very",
        lemma="很",
        pos="adverb",
        gloss="very",
        pronunciation="hěn",
        frequency_rank=16,
    )

    normalized = normalize_callback_lesson(
        unresolved,
        "zh-Hans",
        fallback_terms=[known],
    )

    assert normalized.term_catalog()["hen-very"] == known
    assert normalized.expand().blocks[0].sentences[0].runs[0].term == known


def test_callback_normalization_rebinds_wrong_key_only_for_one_exact_surface_identity() -> None:
    wrong = _term_run("湖", "shui-water", gloss="water")
    wrong["term"].update({"lemma": "水", "pos": "noun", "frequency_rank": 200})
    response = GenerationCallbackResult.model_validate(
        {"schema_version": 1, "lessons": [_chinese_generated_draft([wrong])]}
    )
    lake = LessonTerm(
        key="hu-lake",
        lemma="湖",
        pos="noun",
        gloss="lake",
        pronunciation="hú",
        frequency_rank=900,
    )

    normalized = normalize_callback_lesson(
        response.lessons[0],
        "zh-Hans",
        fallback_terms=[lake],
    )

    run = normalized.blocks[0].sentences[0].runs[0]
    assert (run.text, run.term_key) == ("湖", "hu-lake")
    assert normalized.term_catalog()["hu-lake"] == lake
    assert generated_unit_errors(normalized.expand(), "zh-Hans") == []

    contextual_wire = response.model_dump(mode="json")
    contextual_wire["lessons"][0]["blocks"][0]["sentences"][0]["runs"][0]["pronunciation"] = "hú"
    contextual = GenerationCallbackResult.model_validate(contextual_wire).lessons[0]
    contextual_normalized = normalize_callback_lesson(
        contextual,
        "zh-Hans",
        fallback_terms=[lake],
    )
    contextual_run = contextual_normalized.blocks[0].sentences[0].runs[0]
    assert (contextual_run.term_key, contextual_run.pronunciation) == ("hu-lake", "hú")

    mismatched_wire = response.model_dump(mode="json")
    mismatched_wire["lessons"][0]["blocks"][0]["sentences"][0]["runs"][0]["pronunciation"] = "shuǐ"
    mismatched = GenerationCallbackResult.model_validate(mismatched_wire).lessons[0]
    mismatched_result = normalize_callback_lesson(
        mismatched,
        "zh-Hans",
        fallback_terms=[lake],
    )
    assert mismatched_result.blocks[0].sentences[0].runs == mismatched.blocks[0].sentences[0].runs

    lake_bank = lake.model_copy(update={"key": "hu-bank", "gloss": "lake bank"})
    ambiguous = normalize_callback_lesson(
        response.lessons[0],
        "zh-Hans",
        fallback_terms=[lake, lake_bank],
    )
    ambiguous_run = ambiguous.blocks[0].sentences[0].runs[0]
    assert (ambiguous_run.text, ambiguous_run.term_key) == ("湖", "shui-water")
    assert any(
        "surface text must equal" in error
        for error in generated_unit_errors(ambiguous.expand(), "zh-Hans")
    )


def test_callback_term_catalog_reduces_representative_response_size() -> None:
    payload = _callback_sized_generated_draft(
        [_term_run("hola", "es:hola:WORD")],
        target_length=300,
    )
    expanded = GeneratedLessonDraft.model_validate(payload)
    expanded_json = json.dumps(
        {"schema_version": 1, "lessons": [expanded.model_dump(mode="json")]},
        ensure_ascii=False,
        separators=(",", ":"),
    )

    compact_json = GenerationCallbackResult.model_validate(
        {"schema_version": 1, "lessons": [payload]}
    ).model_dump_json()

    assert len(compact_json) < len(expanded_json) * 0.45


def test_generated_target_declarations_are_advisory() -> None:
    response = GenerationCallbackResult.model_validate(
        {
            "schema_version": 1,
            "lessons": [
                _generated_draft(
                    [_term_run("hola", "es:hola:WORD")],
                    targets=["es:absent:WORD"],
                )
            ],
        }
    )

    assert response.lessons[0].target_term_keys == ["es:absent:WORD"]


def test_worker_reconciles_advisory_targets_against_offer_and_text() -> None:
    draft = GeneratedLessonDraft.model_validate(
        _generated_draft(
            [
                _term_run("alto", "es:alto:WORD"),
                _term_run("bajo", "es:bajo:WORD"),
                _term_run("fuera", "es:fuera:WORD"),
            ],
            targets=["es:missing:WORD", "es:fuera:WORD", "es:bajo:WORD"],
        )
    )
    policy = GenerationTargetPolicy(
        preferred_count=1,
        max_count=3,
        target_text_length=300,
        candidate_term_keys=["es:alto:WORD", "es:bajo:WORD", "es:missing:WORD"],
    )

    assert _reconcile_target_term_keys(draft, policy) == ["es:bajo:WORD"]


def test_worker_does_not_realize_a_teaching_target_from_the_title_alone() -> None:
    payload = _generated_draft([_term_run("texto", "es:texto:WORD")], targets=["es:prueba:NOUN"])
    draft = GeneratedLessonDraft.model_validate(payload)
    policy = GenerationTargetPolicy(
        preferred_count=1,
        max_count=1,
        target_text_length=1,
        candidate_term_keys=["es:prueba:NOUN"],
    )

    assert _reconcile_target_term_keys(draft, policy) == []


def test_worker_restores_known_term_definitions_before_import(
    db: Session, lesson_factory: Any
) -> None:
    import_lesson(db, lesson_factory(targets=[]))
    draft = GeneratedLessonDraft.model_validate(
        _generated_draft([_term_run("mañana", "es:manana:NOUN", frequency_rank=999)])
    )

    replaced = _canonicalize_known_term_definitions(
        db,
        draft,
        learning_language="es-ES",
        translation_language="en",
    )

    term = draft.blocks[0].sentences[0].runs[0].term
    assert replaced == 1
    assert term is not None
    assert (term.lemma, term.pos, term.gloss, term.frequency_rank) == (
        "mañana",
        "NOUN",
        "tomorrow",
        5,
    )


def test_worker_restores_known_title_term_definitions(db: Session, lesson_factory: Any) -> None:
    import_lesson(db, lesson_factory(targets=[]))
    payload = _generated_draft([_term_run("hola", "es:hola:WORD")])
    payload["title"] = "mañana"
    payload["title_sentence"] = {
        "key": "title-sentence",
        "runs": [_term_run("mañana", "es:manana:NOUN", frequency_rank=999)],
        "translation": "Tomorrow",
    }
    draft = GeneratedLessonDraft.model_validate(payload)

    replaced = _canonicalize_known_term_definitions(
        db,
        draft,
        learning_language="es-ES",
        translation_language="en",
    )

    term = draft.title_sentence.runs[0].term
    assert replaced == 1
    assert term is not None
    assert (term.lemma, term.pos, term.gloss, term.frequency_rank) == (
        "mañana",
        "NOUN",
        "tomorrow",
        5,
    )


def test_worker_reuses_exact_definition_alias_and_rewrites_references(
    db: Session, lesson_factory: Any
) -> None:
    import_lesson(db, lesson_factory(targets=[]))
    alias_key = "agent:manana:NOUN"
    alias = _term_run("mañana", alias_key, gloss="tomorrow", frequency_rank=999)
    alias["term"]["pos"] = "NOUN"
    probe_keys = [alias_key, *(f"es:probe-{index}:WORD" for index in range(1, 8))]
    runs = [alias]
    runs.extend(
        _term_run(f"probe{index}", key) for index, key in enumerate(probe_keys[1:], start=1)
    )
    draft = GeneratedLessonDraft.model_validate(
        _generated_draft(
            runs,
            targets=[alias_key],
            calibration={
                "sequence": 1,
                "probes": [
                    {"term_key": key, "difficulty": 0.1 + index * 0.1}
                    for index, key in enumerate(probe_keys)
                ],
            },
        )
    )

    replaced = _canonicalize_known_term_definitions(
        db,
        draft,
        learning_language="es-ES",
        translation_language="en",
    )

    term = draft.blocks[0].sentences[0].runs[0].term
    assert replaced == 1
    assert term is not None
    assert (term.key, term.frequency_rank) == ("es:manana:NOUN", 5)
    assert draft.target_term_keys == ["es:manana:NOUN"]
    assert draft.calibration is not None
    assert draft.calibration.probes[0].term_key == "es:manana:NOUN"
    body_keys = {
        run.term.key
        for block in draft.blocks
        for sentence in block.sentences
        for run in sentence.runs
        if run.term is not None
    }
    assert alias_key not in body_keys


def test_worker_rekeys_a_different_definition_that_collides_with_the_library(
    db: Session, lesson_factory: Any
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    colliding_key = "tamen-they"
    import_lesson(
        db,
        lesson_factory(
            key="stored-human-pronoun",
            learning_language="zh-Hans",
            terms=[(colliding_key, "他们", "他们", "pronoun", "they", 80)],
            targets=[],
        ),
    )

    generated_pronoun = _term_run("它们", colliding_key, gloss="they", frequency_rank=120)
    generated_pronoun["term"]["pos"] = "pronoun"
    probe_runs = [
        _term_run(surface, f"zh:probe-{index}:WORD")
        for index, surface in enumerate("甲乙丙丁戊己庚", start=1)
    ]
    probe_keys = [colliding_key, *(run["term"]["key"] for run in probe_runs)]
    payload = _chinese_generated_draft([generated_pronoun, *probe_runs])
    payload["target_term_keys"] = [colliding_key]
    payload["calibration"] = {
        "sequence": 1,
        "probes": [
            {"term_key": key, "difficulty": 0.1 * index}
            for index, key in enumerate(probe_keys, start=1)
        ],
    }
    first = GeneratedLessonDraft.model_validate(deepcopy(payload))
    repeated = GeneratedLessonDraft.model_validate(deepcopy(payload))

    first_replacements = _canonicalize_known_term_definitions(
        db,
        first,
        learning_language="zh-Hans",
        translation_language="en",
    )
    repeated_replacements = _canonicalize_known_term_definitions(
        db,
        repeated,
        learning_language="zh-Hans",
        translation_language="en",
    )

    first_term = first.blocks[0].sentences[0].runs[0].term
    repeated_term = repeated.blocks[0].sentences[0].runs[0].term
    assert first_replacements == repeated_replacements == 1
    assert first_term is not None and repeated_term is not None
    assert first_term.key == repeated_term.key
    assert first_term.key.startswith(f"{colliding_key}~")
    assert (first_term.lemma, first_term.pos, first_term.gloss) == ("它们", "pronoun", "they")
    assert first.target_term_keys == [first_term.key]
    assert first.calibration is not None
    assert first.calibration.probes[0].term_key == first_term.key
    assert generated_unit_errors(first, "zh-Hans") == []

    stable_key = first_term.key
    import_lesson(
        db,
        LessonDocument(
            key="generated-nonhuman-pronoun",
            title=first.title,
            title_sentence=first.title_sentence,
            learning_language="zh-Hans",
            translation_language="en",
            topic=first.topic,
            level=first.level,
            difficulty=first.difficulty,
            blocks=first.blocks,
            target_term_keys=first.target_term_keys,
            calibration=first.calibration,
        ),
    )
    future = GeneratedLessonDraft.model_validate(deepcopy(payload))

    _canonicalize_known_term_definitions(
        db,
        future,
        learning_language="zh-Hans",
        translation_language="en",
    )

    future_term = future.blocks[0].sentences[0].runs[0].term
    assert future_term is not None
    assert future_term.key == stable_key
    assert future.target_term_keys == [stable_key]
    assert future.calibration is not None
    assert future.calibration.probes[0].term_key == stable_key


def test_worker_rekeys_collisions_with_stored_non_learning_display_terms(
    db: Session, lesson_factory: Any
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    colliding_key = "zh:one-classifier"
    import_lesson(
        db,
        lesson_factory(
            key="stored-productive-span",
            learning_language="zh-Hans",
            terms=[
                (
                    colliding_key,
                    "一个",
                    "一个",
                    "quantity_phrase",
                    "one item",
                    10,
                )
            ],
            targets=[],
        ),
    )
    lexicalized = _term_run("一起", colliding_key, gloss="together", frequency_rank=500)
    lexicalized["term"]["pos"] = "adverb"
    payload = _chinese_generated_draft([lexicalized])
    payload["target_term_keys"] = [colliding_key]
    draft = GeneratedLessonDraft.model_validate(payload)

    replaced = _canonicalize_known_term_definitions(
        db,
        draft,
        learning_language="zh-Hans",
        translation_language="en",
    )

    term = draft.blocks[0].sentences[0].runs[0].term
    assert replaced == 1
    assert term is not None
    assert term.key.startswith(f"{colliding_key}~")
    assert (term.lemma, term.pos, term.gloss) == ("一起", "adverb", "together")
    assert draft.target_term_keys == [term.key]
    assert generated_unit_errors(draft, "zh-Hans") == []
    import_lesson(
        db,
        LessonDocument(
            key="generated-lexicalized-term",
            title=draft.title,
            title_sentence=draft.title_sentence,
            learning_language="zh-Hans",
            translation_language="en",
            topic=draft.topic,
            level=draft.level,
            difficulty=draft.difficulty,
            blocks=draft.blocks,
            target_term_keys=draft.target_term_keys,
        ),
    )


def test_worker_fills_target_shortfall_and_obeys_policy_cap() -> None:
    candidate_runs = [
        _term_run("uno", "es:uno:WORD"),
        _term_run("dos", "es:dos:WORD"),
        _term_run("tres", "es:tres:WORD"),
    ]
    filler = _term_run("relleno", "es:relleno:WORD")
    runs = candidate_runs + [filler] * 298
    sentences = [
        {
            "key": f"sentence-{index}",
            "runs": runs[offset : offset + 100],
            "translation": "Filler.",
        }
        for index, offset in enumerate(range(0, len(runs), 100), start=1)
    ]
    payload = _generated_draft([], targets=["es:uno:WORD", "es:dos:WORD", "es:tres:WORD"])
    payload["blocks"][0]["sentences"] = sentences
    draft = GeneratedLessonDraft.model_validate(payload)
    policy = GenerationTargetPolicy(
        preferred_count=1,
        max_count=2,
        target_text_length=300,
        candidate_term_keys=["es:uno:WORD", "es:dos:WORD", "es:tres:WORD"],
    )

    assert _reconcile_target_term_keys(draft, policy) == ["es:uno:WORD", "es:dos:WORD"]

    no_declared = draft.model_copy(update={"target_term_keys": ["es:missing:WORD"]})
    assert _reconcile_target_term_keys(no_declared, policy) == ["es:uno:WORD"]


def test_generated_terms_require_frequency_rank_but_manual_terms_do_not() -> None:
    run = _term_run("hola", "es:hola:WORD")
    del run["term"]["frequency_rank"]
    payload = _generated_draft([run])

    with pytest.raises(ValidationError, match="frequency_rank"):
        GeneratedLessonDraft.model_validate(payload)

    manual = LessonDocument(
        key="manual-without-frequency-rank",
        title="Manual lesson",
        learning_language="es-ES",
        translation_language="en",
        blocks=payload["blocks"],
    )
    assert manual.term_catalog()["es:hola:WORD"].frequency_rank is None


def test_generated_proper_names_may_omit_frequency_rank() -> None:
    run = _term_run("小林", "zh:xiaolin:PROPN")
    run["term"]["pos"] = "proper noun"
    del run["term"]["frequency_rank"]

    draft = GeneratedLessonDraft.model_validate(_generated_draft([run]))

    assert draft.blocks[0].sentences[0].runs[0].term is not None
    assert draft.blocks[0].sentences[0].runs[0].term.frequency_rank is None


def test_generated_calibration_annotates_non_probe_vocabulary() -> None:
    probe_keys = [f"es:probe-{index}:WORD" for index in range(8)]
    runs: list[dict[str, Any]] = []
    for index, key in enumerate(probe_keys):
        if runs:
            runs.append({"text": " "})
        runs.append(_term_run(f"palabra{index}", key))
    runs.extend([{"text": " "}, _term_run("contexto", "es:contexto:WORD"), {"text": "."}])
    draft = GeneratedLessonDraft.model_validate(
        _generated_draft(
            runs,
            targets=["es:contexto:WORD"],
            calibration={
                "sequence": 1,
                "probes": [
                    {"term_key": key, "difficulty": 0.1 + index * 0.08}
                    for index, key in enumerate(probe_keys)
                ],
            },
        )
    )

    annotated_keys = {
        run.term.key
        for block in draft.blocks
        for sentence in block.sentences
        for run in sentence.runs
        if run.term is not None
    }
    assert len(annotated_keys) == 9
    assert draft.target_term_keys == ["es:contexto:WORD"]
    assert draft.calibration is not None
    assert len(draft.calibration.probes) == 8


def test_generation_response_rejects_conflicting_canonical_term_definitions() -> None:
    first = _generated_draft([_term_run("hola", "es:hola:WORD", gloss="hello")])
    second = _generated_draft([_term_run("hola", "es:hola:WORD", gloss="goodbye")])

    with pytest.raises(ValidationError, match="across generated lessons"):
        GenerationCallbackResult.model_validate({"schema_version": 1, "lessons": [first, second]})


def test_generation_response_rejects_conflicting_title_term_definitions() -> None:
    first = _generated_draft([_term_run("hola", "es:hola:WORD")])
    second = _generated_draft([_term_run("adiós", "es:adios:WORD")])
    second["title_sentence"]["runs"][0]["term"]["gloss"] = "exam"

    with pytest.raises(ValidationError, match="across generated lessons"):
        GenerationCallbackResult.model_validate({"schema_version": 1, "lessons": [first, second]})


def test_priority_terms_keep_revealed_terms_ahead_of_many_unseen(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    terms = [
        (f"es:term-{index}:NOUN", f"término{index}", f"término{index}", "NOUN", "term", index)
        for index in range(1, 26)
    ]
    lesson = import_lesson(db, lesson_factory(terms=terms, targets=[]))
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="revealed-after-many-unseen",
                payload={"term_key": "es:term-25:NOUN"},
            )
        ],
    )

    priority = build_agent_brief(db).priority_terms
    assert priority[0].key == "es:term-25:NOUN"
    assert priority[0].reason in {"due", "fragile"}
    assert len([term for term in priority if term.reason == "unseen"]) == len(priority) - 1
    assert 0 < len(priority) < len(terms)


def test_host_defer_command_holds_agent_work(monkeypatch: pytest.MonkeyPatch) -> None:
    from server.agent_worker import deferred_by_host

    monkeypatch.delenv("ARC_LANG_ADMISSION_COMMAND", raising=False)
    assert deferred_by_host() is False
    monkeypatch.setenv("ARC_LANG_ADMISSION_COMMAND", "true")
    assert deferred_by_host() is False
    monkeypatch.setenv("ARC_LANG_ADMISSION_COMMAND", "false")
    assert deferred_by_host() is True
    monkeypatch.setenv("ARC_LANG_ADMISSION_COMMAND", "/nonexistent/command")
    assert deferred_by_host() is True


@pytest.mark.parametrize("failure", [2, 127, -15, "timeout", "syntax"])
def test_host_admission_errors_wait(monkeypatch: pytest.MonkeyPatch, failure: object) -> None:
    from unittest.mock import Mock

    from server import agent_worker

    monkeypatch.setattr(agent_worker, "_host_admission_cache", None)
    monkeypatch.setenv("ARC_LANG_ADMISSION_COMMAND", "host-admission")
    run = Mock(return_value=subprocess.CompletedProcess([], failure))
    if failure == "timeout":
        run.side_effect = subprocess.TimeoutExpired("host-admission", 30)
    elif failure == "syntax":
        monkeypatch.setenv("ARC_LANG_ADMISSION_COMMAND", "'unterminated")
    monkeypatch.setattr(agent_worker.subprocess, "run", run)
    assert agent_worker.deferred_by_host() is True


def test_host_admission_rechecks_after_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from server import agent_worker

    monkeypatch.setattr(agent_worker, "_host_admission_cache", None)
    monkeypatch.setenv("ARC_LANG_ADMISSION_COMMAND", "host-admission")
    clock = Mock(return_value=100.0)
    run = Mock(side_effect=[subprocess.CompletedProcess([], 3), subprocess.CompletedProcess([], 0)])
    monkeypatch.setattr(agent_worker.time, "monotonic", clock)
    monkeypatch.setattr(agent_worker.subprocess, "run", run)
    assert agent_worker.deferred_by_host() is True
    clock.return_value = 159.9
    assert agent_worker.deferred_by_host() is True
    assert run.call_count == 1
    clock.return_value = 160.0
    assert agent_worker.deferred_by_host() is False
    assert run.call_count == 2


@pytest.mark.parametrize("stage", ["prose", "lexical", "translation", "grammar"])
def test_role_unavailability_retains_task_and_reuses_validated_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    from server.generation_queue import generation_failure_context

    workspace, claimed, payloads = _claimed_staged_task(tmp_path)
    first_stages: list[str] = []

    class UnavailableStage:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            first_stages.append(invocation.request.stage)
            if invocation.request.stage == stage:
                return CallbackResult(
                    exit_code=3, payload="", log="no qualified capacity", deferred=True
                )
            return CallbackResult(
                exit_code=0, payload=_staged_payload(invocation, payloads), log="ok"
            )

    process_generation_task(workspace, claimed, UnavailableStage())
    with session_scope(workspace) as db:
        task = db.get(GenerationTask, claimed.id)
        assert task is not None and task.state == "pending"
        assert task.payload["admission_deferrals"] == 1
        assert not generation_failure_context(task)
        assert claim_generation_task(db) is None  # bounded wait, no busy retry
        monkeypatch.setattr(
            "server.generation_queue.utc_now", lambda: utc_now() + timedelta(minutes=2)
        )
        retry = claim_generation_task(db)
        assert retry is not None and retry.id == claimed.id and retry.attempts == 2
    retry_stages: list[str] = []

    class Available:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            retry_stages.append(invocation.request.stage)
            return CallbackResult(
                exit_code=0, payload=_staged_payload(invocation, payloads), log="ok"
            )

    process_generation_task(workspace, retry, Available())
    with session_scope(workspace) as db:
        task = db.get(GenerationTask, claimed.id)
        assert task is not None and task.state == "completed", task.error if task else None
    if stage != "prose":
        assert "prose" not in retry_stages
    if stage == "grammar":
        assert "lexical" not in retry_stages
    assert (workspace.directory / "agent/jobs" / f"task-{claimed.id}-attempt-1").is_dir()
    assert (workspace.directory / "agent/jobs" / f"task-{claimed.id}-attempt-2").is_dir()


def test_repeated_admission_deferrals_do_not_exhaust_draft_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.generation_queue import defer_generation_task

    workspace, claimed, _ = _claimed_staged_task(tmp_path)
    now = utc_now()
    for count in range(10):
        with session_scope(workspace) as db:
            defer_generation_task(
                db, claimed.id, log_path="test.log", reason="capacity unavailable"
            )
            now += timedelta(minutes=2)
            monkeypatch.setattr("server.generation_queue.utc_now", lambda now=now: now)
            claimed = claim_generation_task(db)
            assert claimed is not None
            assert claimed.payload["admission_deferrals"] == count + 1
    with session_scope(workspace) as db:
        fail_generation_task(db, claimed.id, log_path="test.log", error="[prose] invalid response")
        retry_generation_task(db, claimed.id)
        claimed = claim_generation_task(db)
        assert claimed is not None
        fail_generation_task(db, claimed.id, log_path="test.log", error="[prose] invalid response")
        with pytest.raises(ValueError, match="maximum attempt count"):
            retry_generation_task(db, claimed.id)


@pytest.mark.parametrize("failed,waiting", [("lexical", "translation"), ("translation", "grammar")])
def test_parallel_capacity_wait_does_not_forgive_validated_stage_failure(
    tmp_path: Path,
    failed: str,
    waiting: str,
) -> None:
    workspace, claimed, payloads = _claimed_staged_task(tmp_path)

    class MixedResult:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            stage = invocation.request.stage
            if stage == waiting:
                return CallbackResult(exit_code=3, payload="", log="unavailable", deferred=True)
            if stage == failed:
                return CallbackResult(exit_code=1, payload="", log="real failure")
            return CallbackResult(
                exit_code=0, payload=_staged_payload(invocation, payloads), log="ok"
            )

    process_generation_task(workspace, claimed, MixedResult())
    with session_scope(workspace) as db:
        task = db.get(GenerationTask, claimed.id)
        assert task is not None and task.state == "failed"
        assert not task.payload.get("admission_deferrals")
        assert failed in (task.error or "")
