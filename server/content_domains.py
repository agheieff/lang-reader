"""Subject-domain bookkeeping: the log of refused drafts and the one-time backfill of old texts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, get_args

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.content_planning import SubjectDomainError
from server.generation_usage import task_usage
from server.learning import serialize_lesson
from server.lesson_activity import profile_lessons
from server.lesson_content import lesson_document
from server.models import DomainRejection, GenerationTask, Lesson
from server.profile_state import ensure_profile
from server.schemas import LessonDocument, SubjectDomain
from server.workspaces import Workspace


def record_domain_rejection(db: Session, task: GenerationTask, error: SubjectDomainError) -> None:
    db.add(
        DomainRejection(
            task_id=task.id,
            attempt=task.attempts,
            kind=error.kind,
            declared=error.declared,
            avoid_domains=error.avoid,
            allowed_domains=error.allowed,
        )
    )
    db.commit()


def rejection_report(db: Session, workspace: Workspace) -> dict[str, Any]:
    """Every refused draft with its task's outcome and the tokens the refused call cost."""

    agent_directory = workspace.directory / "agent"
    rejections = []
    for row in db.scalars(select(DomainRejection).order_by(DomainRejection.id)):
        task = db.get(GenerationTask, row.task_id)
        rejections.append(
            {
                "task_id": row.task_id,
                "attempt": row.attempt,
                "kind": row.kind,
                "declared": row.declared,
                "avoid_domains": row.avoid_domains,
                "allowed_domains": row.allowed_domains,
                "created_at": row.created_at.isoformat(),
                "task_state": task.state if task is not None else None,
                "task_attempts": task.attempts if task is not None else None,
                "refused_call_usage": task_usage(agent_directory, row.task_id, row.attempt),
            }
        )
    tasks = db.scalar(select(func.count()).select_from(GenerationTask)) or 0
    return {"tasks": tasks, "rejections": len(rejections), "items": rejections}


def backfill_domains(db: Session, domains: Mapping[str, str]) -> dict[str, list[str]]:
    """Give texts that predate the domain rule a ``content_domain``, keyed by lesson key.

    Only the plan metadata changes, never the text. A text that already has a domain is left
    alone, so a rerun does nothing. The source is recorded because the domain was assigned
    afterwards from the stored prose, not declared by the model that wrote it.
    """

    allowed = set(get_args(SubjectDomain))
    unknown = sorted(key for key, domain in domains.items() if domain not in allowed)
    if unknown:
        raise ValueError(f"unknown subject domain for: {', '.join(unknown)}")
    lessons = {lesson.key: lesson for lesson in profile_lessons(db, ensure_profile(db))}
    missing = sorted(set(domains) - set(lessons))
    if missing:
        raise ValueError(f"unknown lessons: {', '.join(missing)}")
    result: dict[str, list[str]] = {"updated": [], "unchanged": []}
    for key, domain in domains.items():
        lesson = lessons[key]
        document = lesson_document(lesson)
        if document.calibration is not None or document.metadata.get("content_domain"):
            result["unchanged"].append(key)
            continue
        _set_metadata(
            db,
            lesson,
            document,
            {**document.metadata, "content_domain": domain, "content_domain_source": "backfill"},
        )
        result["updated"].append(key)
    db.commit()
    return result


def _set_metadata(
    db: Session, lesson: Lesson, document: LessonDocument, metadata: dict[str, Any]
) -> None:
    updated = document.model_copy(update={"metadata": metadata})
    payload, content_hash = serialize_lesson(updated)
    # updated_at stays: it is the derived-state watermark and the learning evidence is unchanged.
    lesson.payload = payload
    lesson.metadata_json = metadata
    lesson.content_hash = content_hash
