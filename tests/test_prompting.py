from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from server.agent_worker import _stage_request
from server.generation_callback import canonical_request
from server.prompting import build_prompt
from tests.test_generation import _routing_test_request


def _document(prompt: str) -> dict[str, Any]:
    document = json.loads(prompt.split("in this request:\n", 1)[1])
    assert isinstance(document, dict)
    return document


def test_prose_prompt_is_smaller_and_static_first(db: Session, lesson_factory: Any) -> None:
    request = _stage_request(_routing_test_request(db, lesson_factory), "prose")
    prompt = build_prompt(request)
    document = _document(prompt)

    assert len(prompt) < len(canonical_request(request))
    keys = list(document)
    assert keys[:2] == ["stage", "schema_version"] and keys[2] == "instructions"
    assert keys.index("instructions") < keys.index("brief") < keys.index("target_policy")
    for host_key in ("job_id", "task_id", "attempt", "profile_fingerprint", "workspace_path"):
        assert host_key not in document
    # Prose chooses from priority_grammar; the full catalog is for the annotation stages.
    assert "grammar_catalog" not in document["brief"]
    assert "generated_at" not in document["brief"]


def test_calibration_prompt_keeps_the_grammar_catalog(db: Session, lesson_factory: Any) -> None:
    base = _routing_test_request(db, lesson_factory)
    legacy = _document(build_prompt(base))
    assert "grammar_catalog" in legacy["brief"]
    assert "Generate complete lesson drafts" in build_prompt(base)


def test_recent_lessons_keep_only_what_variety_needs(db: Session, lesson_factory: Any) -> None:
    request = _stage_request(_routing_test_request(db, lesson_factory), "prose")
    recent = _document(build_prompt(request))["brief"]["recent_lessons"]
    assert recent
    for lesson in recent:
        assert set(lesson) <= {
            "title",
            "topic",
            "opening_excerpt",
            "ending_excerpt",
            "completed",
            "rating",
            "feedback",
            "grammar_keys",
            "calibration_lesson",
            "content_plan",
        }


def test_prompt_never_contains_the_response_schema_or_secrets(
    db: Session, lesson_factory: Any
) -> None:
    prompt = build_prompt(_stage_request(_routing_test_request(db, lesson_factory), "prose"))
    assert "Return only JSON matching the supplied output schema" in prompt
