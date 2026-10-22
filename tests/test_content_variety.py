"""Pins the variety behaviour: recent subject domains are avoided and enforced, and structure
rotates, even when the model keeps preferring the same few favourites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.agent_brief import build_agent_brief
from server.agent_worker import (
    _maintain_worker_workspaces,
    _prose_stage_instructions,
    _stage_request,
    process_generation_task,
)
from server.content_domains import backfill_domains, rejection_report
from server.content_planning import (
    DOMAIN_WINDOW,
    SUBJECT_DOMAINS,
    build_content_plan,
    content_plan_instructions,
    subject_domain_violation,
)
from server.db import init_db, session_scope
from server.generation_callback import CallbackInvocation, CallbackResult
from server.learning import import_lesson, record_events, serialize_lesson
from server.lesson_content import lesson_document
from server.models import GenerationTask, Lesson
from server.preference_agent import maybe_update_preferences
from server.profile_activation import ProfileActivationUpdate, activate_profile_settings
from server.prompting import build_prompt
from server.reading_preferences import content_history, preferences_view, update_due
from server.schemas import GenerationProseResult, PreferenceUpdateRequest
from server.workspaces import Workspace, WorkspaceRegistry
from tests.test_generation import (
    _claimed_staged_task,
    _routing_test_request,
    _staged_payload,
)


def _import(db: Session, lesson_factory: Any, key: str, domain: str | None, plan: Any) -> None:
    metadata: dict[str, Any] = {"content_plan": plan.model_dump(mode="json")}
    if domain is not None:
        metadata["content_domain"] = domain
    import_lesson(db, lesson_factory(key=key, metadata=metadata))


def test_history_carries_the_domain_and_the_plan_avoids_recent_ones(
    db: Session, lesson_factory: Any
) -> None:
    plan = build_content_plan(build_agent_brief(db), task_id=1, requested_topic=None)
    assert plan.avoid_domains == []
    _import(db, lesson_factory, "a", "technology-devices", plan)
    _import(db, lesson_factory, "b", "food-cooking", plan)
    _import(db, lesson_factory, "legacy", None, plan)

    assert [item.domain for item in content_history(db)] == [
        None,
        "food-cooking",
        "technology-devices",
    ]
    brief = build_agent_brief(db)
    avoided = build_content_plan(brief, task_id=2, requested_topic=None).avoid_domains
    assert avoided == ["food-cooking", "technology-devices"]
    # A requested topic or an explicit "more of this" may stay in a recent domain.
    assert build_content_plan(brief, task_id=2, requested_topic="radios").avoid_domains == []
    assert (
        build_content_plan(
            brief, task_id=2, requested_topic=None, feedback=("same_topic",)
        ).avoid_domains
        == []
    )
    instructions = content_plan_instructions()
    assert "subject_domain" in instructions and "avoid_domains" in instructions


def test_a_draft_repeating_a_recent_domain_or_omitting_it_is_rejected(
    db: Session, lesson_factory: Any
) -> None:
    plan = build_content_plan(build_agent_brief(db), task_id=1, requested_topic=None)
    _import(db, lesson_factory, "a", "technology-devices", plan)
    plan = build_content_plan(build_agent_brief(db), task_id=2, requested_topic=None)

    assert subject_domain_violation(plan, ["music"]) is None
    repeated = subject_domain_violation(plan, ["technology-devices"])
    assert repeated is not None and repeated.kind == "repeat"
    assert "technology-devices" in str(repeated)
    assert "music" in repeated.allowed and "technology-devices" not in repeated.allowed
    missing = subject_domain_violation(plan, [None])
    assert missing is not None and missing.kind == "missing"
    duplicate = subject_domain_violation(plan, ["music", "music"])
    assert duplicate is not None and duplicate.kind == "duplicate"


def test_replay_of_a_model_that_keeps_choosing_the_same_favourites(
    db: Session, lesson_factory: Any
) -> None:
    favourites = ["technology-devices", "science-space", "weather-seasons", "crafts-repair"]
    domains: list[str] = []
    archetypes: list[str] = []
    rejected = 0
    for task_id in range(1, 41):
        plan = build_content_plan(build_agent_brief(db), task_id=task_id, requested_topic=None)
        choice = next(
            (domain for domain in favourites if domain not in plan.avoid_domains), None
        ) or next(domain for domain in SUBJECT_DOMAINS if domain not in plan.avoid_domains)
        # The model's first instinct is always its top favourite; the host then rejects it.
        if subject_domain_violation(plan, [favourites[0]]) is not None:
            rejected += 1
        assert subject_domain_violation(plan, [choice]) is None
        domains.append(choice)
        archetypes.append(plan.archetype)
        _import(db, lesson_factory, f"t{task_id}", choice, plan)

    for index, domain in enumerate(domains):
        assert domain not in domains[max(0, index - DOMAIN_WINDOW) : index]
    assert rejected >= 20
    assert len(set(domains)) >= DOMAIN_WINDOW + 1
    # The favourites are not starved: the first one comes back as soon as the window allows.
    assert domains.count(favourites[0]) >= len(domains) // (DOMAIN_WINDOW + 1) - 1
    for index in range(len(archetypes)):
        assert archetypes[index] not in archetypes[max(0, index - 4) : index]


def _run_with_prose_domain(
    tmp_path: Path, lesson_factory: Any, domain: str
) -> tuple[Workspace, GenerationTask, list[Lesson]]:
    workspace, claimed, payloads = _claimed_staged_task(tmp_path)
    with session_scope(workspace) as db:
        earlier = build_content_plan(build_agent_brief(db), task_id=1, requested_topic=None)
        _import(db, lesson_factory, "earlier", "music", earlier)
    prose = json.loads(payloads["prose"])
    for lesson in prose["lessons"]:
        lesson["subject_domain"] = domain
    payloads["prose"] = json.dumps(prose)

    class Callback:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            payload = _staged_payload(invocation, payloads)
            return CallbackResult(exit_code=0, payload=payload, log="ok")

    process_generation_task(workspace, claimed, Callback())
    with session_scope(workspace) as db:
        task = db.get(GenerationTask, claimed.id)
        assert task is not None
        lessons = list(db.scalars(select(Lesson).where(Lesson.key != "earlier")))
        db.expunge_all()
    return workspace, task, lessons


def test_worker_rejects_a_prose_draft_in_a_recent_domain_and_records_a_fresh_one(
    tmp_path: Path, lesson_factory: Any
) -> None:
    workspace, task, lessons = _run_with_prose_domain(tmp_path / "repeat", lesson_factory, "music")
    assert task.state == "failed" and not lessons
    assert task.error is not None and "repeats the recent subject domain 'music'" in task.error
    with session_scope(workspace) as db:
        report = rejection_report(db, workspace)
    assert report["rejections"] == 1 and report["tasks"] == 1
    [item] = report["items"]
    assert (item["task_id"], item["attempt"], item["kind"]) == (task.id, 1, "repeat")
    assert item["declared"] == ["music"] and item["avoid_domains"] == ["music"]
    assert "music" not in item["allowed_domains"] and "history" in item["allowed_domains"]
    assert item["task_state"] == "failed" and item["refused_call_usage"]["calls"] == 1

    workspace, task, lessons = _run_with_prose_domain(tmp_path / "fresh", lesson_factory, "history")
    assert task.state == "completed", task.error
    assert [lesson.metadata_json["content_domain"] for lesson in lessons] == ["history"]
    assert lessons[0].metadata_json["content_plan"]["avoid_domains"] == ["music"]
    with session_scope(workspace) as db:
        assert rejection_report(db, workspace)["rejections"] == 0


def test_backfill_assigns_domains_without_touching_the_text(
    db: Session, lesson_factory: Any
) -> None:
    old = import_lesson(db, lesson_factory(key="old"))
    declared = import_lesson(
        db, lesson_factory(key="declared", metadata={"content_domain": "music"})
    )
    before = lesson_document(old)

    result = backfill_domains(db, {"old": "history", "declared": "sport-fitness"})

    assert result == {"updated": ["old"], "unchanged": ["declared"]}
    db.refresh(old)
    after = lesson_document(old)
    assert after.metadata["content_domain"] == "history"
    assert after.metadata["content_domain_source"] == "backfill"
    assert old.metadata_json == after.metadata
    assert after.blocks == before.blocks and after.title == before.title
    assert old.content_hash == serialize_lesson(after)[1]
    assert [item.domain for item in content_history(db)] == ["music", "history"]
    assert db.refresh(declared) is None and declared.metadata_json["content_domain"] == "music"
    assert backfill_domains(db, {"old": "history"}) == {"updated": [], "unchanged": ["old"]}
    with pytest.raises(ValueError, match="unknown subject domain"):
        backfill_domains(db, {"old": "gardening"})
    with pytest.raises(ValueError, match="unknown lessons"):
        backfill_domains(db, {"nope": "history"})


def test_forced_refresh_runs_the_normal_update_before_three_reactions(tmp_path: Path) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    seen: list[PreferenceUpdateRequest] = []

    class Agent:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            assert isinstance(invocation.request, PreferenceUpdateRequest)
            seen.append(invocation.request)
            payload = {"schema_version": 1, "preferences": "Likes: sport", "reason": "refresh"}
            return CallbackResult(exit_code=0, payload=json.dumps(payload), log="")

    with session_scope(workspace) as db:
        activate_profile_settings(db, ProfileActivationUpdate())
        assert not update_due(db)
    assert maybe_update_preferences(workspace, Agent()) is False and not seen
    assert maybe_update_preferences(workspace, Agent(), force=True) is True
    assert seen[0].user_messages == []
    with session_scope(workspace) as db:
        assert preferences_view(db).last_agent_reason == "refresh"


def test_prose_prompt_and_schema_carry_the_domain_rule(db: Session, lesson_factory: Any) -> None:
    request = _stage_request(_routing_test_request(db, lesson_factory), "prose")
    request = request.model_copy(
        update={
            "content_plan": request.content_plan.model_copy(update={"avoid_domains": ["music"]})
        }
    )
    document = json.loads(build_prompt(request).split("in this request:\n", 1)[1])
    assert document["content_plan"]["avoid_domains"] == ["music"]
    assert "subject_domain" in _prose_stage_instructions(request)
    schema = json.dumps(GenerationProseResult.model_json_schema())
    assert '"subject_domain"' in schema and "technology-devices" in schema


def test_three_reactions_make_the_worker_update_the_notes_with_domains(
    tmp_path: Path, lesson_factory: Any, event_factory: Any
) -> None:
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    init_db(workspace)
    seen: list[PreferenceUpdateRequest] = []

    class Agent:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            assert isinstance(invocation.request, PreferenceUpdateRequest)
            seen.append(invocation.request)
            payload = {"schema_version": 1, "preferences": "Likes: music", "reason": "3 reactions"}
            return CallbackResult(exit_code=0, payload=json.dumps(payload), log="")

    with session_scope(workspace) as db:
        activate_profile_settings(db, ProfileActivationUpdate())
        plan = build_content_plan(build_agent_brief(db), task_id=1, requested_topic=None)
        lessons = [
            import_lesson(
                db,
                lesson_factory(
                    key=f"t{n}",
                    metadata={
                        "content_plan": plan.model_dump(mode="json"),
                        "content_domain": "music",
                    },
                ),
            )
            for n in range(3)
        ]

        def completed(lesson_id: int, n: int) -> dict[str, Any]:
            return event_factory(
                lesson_id,
                "lesson.completed",
                event_id=f"done-{n}",
                payload={"active_seconds": 60, "completion_ratio": 1},
            )

        assert not update_due(db)
        record_events(db, [completed(lesson.id, n) for n, lesson in enumerate(lessons[:2])])
        assert not update_due(db)
        record_events(db, [completed(lessons[2].id, 2)])
        assert update_due(db)

    _maintain_worker_workspaces([workspace], Agent())

    assert [item.domain for item in seen[0].content_history] == ["music"] * 3
    with session_scope(workspace) as db:
        assert preferences_view(db).source == "agent" and not update_due(db)


def test_cli_commands_backfill_report_and_refresh(
    tmp_path: Path,
    lesson_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from server import agent_worker, cli, model_callback

    registry = WorkspaceRegistry(tmp_path)
    monkeypatch.setattr(cli, "registry", registry)
    workspace = registry.resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        activate_profile_settings(db, ProfileActivationUpdate())
        import_lesson(db, lesson_factory(key="old"))
    domains = tmp_path / "domains.json"
    domains.write_text(json.dumps({"old": "history"}), encoding="utf-8")

    def run(*argv: str) -> Any:
        args = cli.build_parser().parse_args(argv)
        args.handler(args)
        return json.loads(capsys.readouterr().out)

    assert run("content", "backfill-domains", str(domains)) == {"updated": ["old"], "unchanged": []}
    assert run("generation", "rejections") == {"tasks": 0, "rejections": 0, "items": []}

    class Agent:
        def run(self, invocation: CallbackInvocation) -> CallbackResult:
            payload = {"schema_version": 1, "preferences": "Likes: history", "reason": "forced"}
            return CallbackResult(exit_code=0, payload=json.dumps(payload), log="")

    monkeypatch.setattr(model_callback, "load_callback", lambda: Agent())
    monkeypatch.setattr(agent_worker, "deferred_by_host", lambda: True)
    with pytest.raises(ValueError, match="deferred"):
        run("preferences", "refresh")
    monkeypatch.setattr(agent_worker, "deferred_by_host", lambda: False)
    refreshed = run("preferences", "refresh")
    assert refreshed["before"]["source"] == "default"
    assert (
        refreshed["after"]["source"] == "agent" and refreshed["after"]["text"] == "Likes: history\n"
    )
