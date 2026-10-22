"""Read-only text-library projections for one language profile."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.clock import as_utc
from server.generation_tasks import (
    generated_lesson_key,
    generation_task_kind,
    requested_topic,
    topic_request_id,
    waiting_for_model_capacity,
)
from server.grammar import comfortable_grammar_construction_keys
from server.grammar_catalog import find_grammar_catalog
from server.learning import build_reader_term_bands
from server.lesson_activity import LessonActivity, lesson_activities
from server.lesson_content import lesson_document, lexical_token_count
from server.lesson_queue import ordered_profile_lessons
from server.models import GenerationTask, Lesson
from server.profile_state import build_profile_view, ensure_profile
from server.schemas import (
    AgentGrammarCatalogEntry,
    TextDetail,
    TextPreparationView,
    TextRequestView,
    TextsState,
    TextView,
)
from server.vocabulary_plan import KnownShareInputs, known_share, profile_known_share_inputs


def get_texts_state(db: Session) -> TextsState:
    """List imported texts and durable user-requested generation work without side effects."""

    profile = ensure_profile(db)
    lessons = ordered_profile_lessons(db)
    lesson_ids = [lesson.id for lesson in lessons]
    activity = lesson_activities(db, lesson_ids)
    ready_ids = [lesson.id for lesson in lessons if activity[lesson.id].ready]
    queue_positions = {lesson_id: index for index, lesson_id in enumerate(ready_ids, 1)}
    share_inputs = profile_known_share_inputs(db)
    texts = [
        _text_view(
            lesson,
            activity[lesson.id],
            queue_positions.get(lesson.id),
            share_inputs,
        )
        for lesson in lessons
    ]
    tasks = db.scalars(
        select(GenerationTask).order_by(GenerationTask.created_at.desc(), GenerationTask.id.desc())
    ).all()
    active_tasks = sorted(
        (
            task
            for task in tasks
            if task.state in {"pending", "running"} and not _task_lesson_is_imported(db, task)
        ),
        key=lambda task: (
            task.state != "running",
            as_utc(task.created_at),
            task.id,
        ),
    )
    preparations = [text_preparation_view(task) for task in active_tasks]
    requests = [
        text_request_view(db, task)
        for task in tasks
        if generation_task_kind(task) == "topic_request"
    ]
    return TextsState(
        learning_language=profile.learning_language,
        translation_language=profile.translation_language,
        texts=texts,
        preparations=preparations,
        requests=requests,
    )


def get_text_detail(db: Session, lesson_id: int) -> TextDetail:
    """Return a lesson for previewing without selecting or opening a reading session."""

    profile = ensure_profile(db)
    lesson = db.scalar(
        select(Lesson).where(
            Lesson.id == lesson_id,
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
        )
    )
    if lesson is None:
        raise LookupError(f"lesson not found: {lesson_id}")
    document = lesson_document(lesson)
    used_grammar = document.grammar_construction_keys()
    catalog = find_grammar_catalog(document.learning_language)
    return TextDetail(
        lesson_id=lesson.id,
        lesson=document,
        term_bands=build_reader_term_bands(db, document, build_profile_view(db, profile)),
        grammar_catalog=[
            AgentGrammarCatalogEntry.model_validate(construction.model_dump())
            for construction in (catalog.constructions if catalog is not None else ())
            if construction.key in used_grammar
        ],
        comfortable_grammar_construction_keys=comfortable_grammar_construction_keys(
            db,
            learning_language=document.learning_language,
            translation_language=document.translation_language,
            construction_keys=used_grammar,
        ),
    )


def text_request_view(db: Session, task: GenerationTask) -> TextRequestView:
    if generation_task_kind(task) != "topic_request":
        raise ValueError("generation task is not a topic request")
    request_id = topic_request_id(task)
    if request_id is None:
        raise ValueError("topic request task has invalid request data")
    lesson_id = db.scalar(
        select(Lesson.id).where(Lesson.key == generated_lesson_key(task.id, 1)).limit(1)
    )
    return TextRequestView(
        task_id=task.id,
        request_id=request_id,
        topic=requested_topic(task),
        state=task.state,
        created_at=as_utc(task.created_at),
        updated_at=as_utc(task.updated_at),
        error=task.error,
        lesson_id=lesson_id,
    )


def text_preparation_view(task: GenerationTask) -> TextPreparationView:
    kind = generation_task_kind(task)
    if task.state not in {"pending", "running"}:
        raise ValueError("generation task is not an active text preparation")
    topic = requested_topic(task)
    return TextPreparationView(
        task_id=task.id,
        state=task.state,
        request_kind=kind,
        requested_topic=topic,
        waiting_for_capacity=waiting_for_model_capacity(task),
        created_at=as_utc(task.created_at),
        updated_at=as_utc(task.updated_at),
    )


def _task_lesson_is_imported(db: Session, task: GenerationTask) -> bool:
    lesson_key = generated_lesson_key(task.id, 1)
    return db.scalar(select(Lesson.id).where(Lesson.key == lesson_key).limit(1)) is not None


def _text_view(
    lesson: Lesson,
    activity: LessonActivity,
    queue_position: int | None,
    share_inputs: KnownShareInputs,
) -> TextView:
    document = lesson_document(lesson)
    return TextView(
        id=lesson.id,
        key=lesson.key,
        title=document.title,
        topic=document.topic,
        level=document.level,
        difficulty=document.difficulty,
        imported_at=as_utc(lesson.imported_at),
        status=activity.status,
        queue_position=queue_position,
        opened_at=activity.opened_at,
        last_completed_at=activity.last_completed_at,
        skipped_at=activity.skipped_at,
        session_count=len(activity.session_ids),
        completion_count=len(activity.completion_session_ids),
        rating=activity.rating,
        lexical_token_count=lexical_token_count(document),
        known_share=known_share(document, share_inputs),
    )
