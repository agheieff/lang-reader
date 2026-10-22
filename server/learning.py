"""Lesson import, reader state, and event recording."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from server.agent_brief import agent_grammar_catalog_entry
from server.clock import as_utc, utc_now
from server.derived_state import ensure_derived_states
from server.generation_queue import (
    GENERATION_TRIGGER_EVENT_TYPES,
    ensure_generation_task,
    queue_generation_status,
)
from server.grammar import comfortable_grammar_construction_keys
from server.grammar_catalog import (
    find_grammar_catalog,
)
from server.learning_policy import TERM_BAND_POLICY, TermBandPolicy, frontier_frequency_rank
from server.learning_units import LearningUnitIndex, build_learning_unit_index
from server.lesson_content import lesson_document
from server.lesson_queue import (
    unread_lessons,
)
from server.models import (
    Interaction,
    Lesson,
    LessonQueueAction,
    LessonQueueMoveAction,
    LexemeState,
)
from server.profile_activation import (
    profile_is_active,
)
from server.profile_fingerprint import canonical_json
from server.profile_state import build_profile_view, ensure_profile
from server.schemas import (
    EventRecordResult,
    FeedbackTag,
    InteractionIn,
    LessonDocument,
    ProfileView,
    ReaderProgress,
    ReaderState,
    TermKnowledgeBand,
    is_proper_noun_pos,
)
from server.session_feedback import feedback_from_payload
from server.vocabulary_plan import (
    known_share,
    profile_known_share_inputs,
)

DERIVED_EVIDENCE_EVENT_TYPES = frozenset(
    {"term.revealed", "translation.revealed", "lesson.completed"}
)
PROFICIENCY_EVENT_TYPES = DERIVED_EVIDENCE_EVENT_TYPES | {"lesson.rated"}


class LessonConflictError(ValueError):
    pass


class EventConflictError(ValueError):
    pass


def validate_lesson(payload: LessonDocument | Mapping[str, Any]) -> LessonDocument:
    if isinstance(payload, LessonDocument):
        return payload
    return LessonDocument.model_validate(payload)


def serialize_lesson(document: LessonDocument) -> tuple[dict[str, Any], str]:
    """The stored payload of a lesson and the content hash that identifies it."""

    serialized = document.model_dump(mode="json")
    if serialized["calibration"] is None:
        del serialized["calibration"]
    canonical = json.dumps(serialized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return serialized, hashlib.sha256(canonical.encode()).hexdigest()


def import_lesson(
    db: Session,
    payload: LessonDocument | Mapping[str, Any],
    *,
    source_path: str | None = None,
    replace: bool = False,
) -> Lesson:
    document = validate_lesson(payload)
    profile = ensure_profile(db)
    if (
        document.learning_language != profile.learning_language
        or document.translation_language != profile.translation_language
    ):
        raise ValueError("lesson languages must match the local profile")

    _validate_term_keys_across_lessons(db, document)
    serialized, content_hash = serialize_lesson(document)
    existing = db.scalar(select(Lesson).where(Lesson.key == document.key))

    if existing is not None:
        if existing.content_hash == content_hash:
            _ensure_lesson_audio(db, existing)
            return existing
        if not replace:
            raise LessonConflictError(
                f"lesson already exists with different content: {document.key}"
            )
        has_events = db.scalar(
            select(func.count())
            .select_from(Interaction)
            .where(Interaction.lesson_id == existing.id)
        )
        if has_events:
            raise LessonConflictError("cannot replace a lesson after interactions were recorded")
        has_queue_actions = db.scalar(
            select(func.count())
            .select_from(LessonQueueAction)
            .where(LessonQueueAction.lesson_id == existing.id)
        )
        if has_queue_actions:
            raise LessonConflictError("cannot replace a lesson after queue actions were recorded")
        has_queue_moves = db.scalar(
            select(func.count())
            .select_from(LessonQueueMoveAction)
            .where(
                or_(
                    LessonQueueMoveAction.lesson_id == existing.id,
                    LessonQueueMoveAction.neighbor_lesson_id == existing.id,
                )
            )
        )
        if has_queue_moves:
            raise LessonConflictError("cannot replace a lesson after queue moves were recorded")
        lesson = existing
    else:
        lesson = Lesson(key=document.key)
        db.add(lesson)

    lesson.schema_version = document.schema_version
    lesson.title = document.title
    lesson.learning_language = document.learning_language
    lesson.translation_language = document.translation_language
    lesson.topic = document.topic
    lesson.level = document.level
    lesson.difficulty = document.difficulty
    lesson.payload = serialized
    lesson.metadata_json = document.metadata
    lesson.content_hash = content_hash
    lesson.source_path = source_path
    lesson.updated_at = utc_now()
    db.commit()
    db.refresh(lesson)
    ensure_derived_states(db)
    if lesson.known_share_at_import is None:
        # Keep updated_at unchanged: it is part of the derived-state watermark.
        db.execute(
            update(Lesson)
            .where(Lesson.id == lesson.id)
            .values(
                known_share_at_import=known_share(document, profile_known_share_inputs(db)),
                updated_at=lesson.updated_at,
            )
        )
        db.commit()
        db.refresh(lesson)
    _ensure_lesson_audio(db, lesson)
    return lesson


def _ensure_lesson_audio(db: Session, lesson: Lesson) -> None:
    # The optional provider is isolated, but its small durable queue belongs to every import path.
    from server.tts import ensure_lesson_audio_task, model_language

    profile = ensure_profile(db)
    if profile_is_active(profile) and model_language(lesson.learning_language) is not None:
        ensure_lesson_audio_task(db, lesson)


def get_reader_state(
    db: Session, lesson_id: int | None = None, *, fresh: bool = False
) -> ReaderState:
    profile = ensure_profile(db)
    if lesson_id is None:
        lesson = next(iter(unread_lessons(db)), None)
    else:
        lesson = db.scalar(
            select(Lesson).where(
                Lesson.id == lesson_id,
                Lesson.learning_language == profile.learning_language,
                Lesson.translation_language == profile.translation_language,
            )
        )
        if lesson is None:
            raise LookupError(f"lesson not found: {lesson_id}")

    profile_view = build_profile_view(db, profile)
    generation_status = queue_generation_status(db)
    if lesson is None:
        return ReaderState(profile=profile_view, generation_status=generation_status)

    progress = ReaderProgress()
    latest = None
    if not fresh:
        latest = db.scalar(
            select(Interaction)
            .where(Interaction.lesson_id == lesson.id)
            .order_by(Interaction.occurred_at.desc(), Interaction.id.desc())
            .limit(1)
        )
    if latest is not None:
        events = db.scalars(
            select(Interaction)
            .where(
                Interaction.lesson_id == lesson.id,
                Interaction.session_id == latest.session_id,
            )
            .order_by(Interaction.occurred_at, Interaction.id)
        ).all()
        progress = _reader_progress(latest.session_id, events)

    document = lesson_document(lesson)
    catalog = find_grammar_catalog(document.learning_language)
    used_grammar = document.grammar_construction_keys()
    return ReaderState(
        profile=profile_view,
        lesson_id=lesson.id,
        lesson=document,
        progress=progress,
        term_bands=build_reader_term_bands(db, document, profile_view),
        grammar_catalog=[
            agent_grammar_catalog_entry(construction)
            for construction in (catalog.constructions if catalog is not None else ())
            if construction.key in used_grammar
        ],
        comfortable_grammar_construction_keys=comfortable_grammar_construction_keys(
            db,
            learning_language=document.learning_language,
            translation_language=document.translation_language,
            construction_keys=used_grammar,
        ),
        generation_status=generation_status,
    )


def _profile_learning_unit_index(
    db: Session,
    *,
    document: LessonDocument,
) -> LearningUnitIndex:
    """Build a stable profile index, replacing any stored copy of ``document``."""

    lessons = db.scalars(
        select(Lesson)
        .where(
            Lesson.learning_language == document.learning_language,
            Lesson.translation_language == document.translation_language,
            Lesson.key != document.key,
        )
        .order_by(Lesson.imported_at, Lesson.id)
    ).all()
    documents = [lesson_document(lesson) for lesson in lessons]
    documents.append(document)
    return build_learning_unit_index(documents)


def build_reader_term_bands(
    db: Session,
    document: LessonDocument,
    profile: ProfileView,
    *,
    policy: TermBandPolicy = TERM_BAND_POLICY,
) -> dict[str, TermKnowledgeBand]:
    """Classify current lesson terms for styling without treating a band as evidence."""

    catalog = document.term_catalog()
    units = _profile_learning_unit_index(db, document=document)
    canonical_keys = {key for key in (units.resolve(key) for key in catalog) if key is not None}
    states = {
        state.term_key: state
        for state in db.scalars(
            select(LexemeState).where(
                LexemeState.learning_language == document.learning_language,
                LexemeState.translation_language == document.translation_language,
                LexemeState.term_key.in_(canonical_keys),
            )
        )
    }
    expected_rank = _expected_frequency_rank(profile, policy)
    targets = {
        key
        for key in (units.resolve(target) for target in document.target_term_keys)
        if key is not None
    }
    bands: dict[str, TermKnowledgeBand] = {}
    for display_key, display_term in catalog.items():
        canonical = units.resolve(display_key)
        if canonical is None:
            bands[display_key] = "incidental"
            continue
        term = units.definitions.get(canonical, display_term)
        bands[display_key] = _reader_term_band(
            states.get(canonical),
            frequency_rank=term.frequency_rank,
            part_of_speech=term.pos,
            deliberate_target=canonical in targets,
            expected_rank=expected_rank,
            policy=policy,
        )
    return bands


def _reader_term_band(
    state: LexemeState | None,
    *,
    frequency_rank: int | None,
    part_of_speech: str,
    deliberate_target: bool,
    expected_rank: int,
    policy: TermBandPolicy,
) -> TermKnowledgeBand:
    if state is not None and _is_familiar(state, policy):
        return "familiar"
    if deliberate_target:
        return "focus"
    if state is not None and state.reveal_failures >= policy.repeated_reveal_threshold:
        return "uncertain"
    if state is not None and (
        state.qualified_exposures >= policy.expected_min_exposures
        and state.mastery >= policy.expected_min_mastery
    ):
        return "expected"
    if frequency_rank is None:
        if is_proper_noun_pos(part_of_speech):
            return "incidental"
        return "uncertain"
    if frequency_rank <= expected_rank:
        return "expected"
    if frequency_rank <= expected_rank * policy.uncertain_frequency_multiplier:
        return "uncertain"
    return "incidental"


def _is_familiar(state: LexemeState, policy: TermBandPolicy) -> bool:
    if (
        state.reveal_failures == 0
        and state.qualified_exposures >= policy.familiar_min_exposures
        and state.mastery >= policy.familiar_min_mastery
    ):
        return True
    return (
        state.qualified_exposures >= policy.familiar_with_failures_min_exposures
        and state.mastery >= policy.familiar_with_failures_min_mastery
    )


def _expected_frequency_rank(profile: ProfileView, policy: TermBandPolicy) -> int:
    return frontier_frequency_rank(profile.difficulty, profile.proficiency.lower, policy)


def record_events(
    db: Session,
    events: Sequence[InteractionIn | Mapping[str, Any]],
    *,
    rebuild_derived: bool = True,
) -> EventRecordResult:
    """Append events; the HTTP path defers the derived-cache replay off the request."""

    parsed = [
        event if isinstance(event, InteractionIn) else InteractionIn.model_validate(event)
        for event in events
    ]
    if not parsed:
        return EventRecordResult(accepted=0, duplicates=0, accepted_event_ids=[])

    profile = ensure_profile(db)
    lesson_ids = {event.lesson_id for event in parsed}
    lessons = {
        lesson.id: lesson
        for lesson in db.scalars(
            select(Lesson).where(
                Lesson.id.in_(lesson_ids),
                Lesson.learning_language == profile.learning_language,
                Lesson.translation_language == profile.translation_language,
            )
        ).all()
    }
    missing = lesson_ids - lessons.keys()
    if missing:
        raise LookupError(f"lessons not found: {sorted(missing)}")

    unique_events: dict[str, InteractionIn] = {}
    duplicates = 0
    for event in parsed:
        previous = unique_events.get(event.event_id)
        if previous is None:
            unique_events[event.event_id] = event
        elif _input_event_fingerprint(previous) != _input_event_fingerprint(event):
            raise EventConflictError(f"event_id reused with different content: {event.event_id}")
        else:
            duplicates += 1

    existing = {
        event.event_id: event
        for event in db.scalars(
            select(Interaction).where(Interaction.event_id.in_(unique_events))
        ).all()
    }
    for event_id, stored in existing.items():
        incoming = unique_events[event_id]
        if _stored_event_fingerprint(stored) != _input_event_fingerprint(incoming):
            raise EventConflictError(f"event_id reused with different content: {event_id}")
        duplicates += 1

    new_events = [event for event in unique_events.values() if event.event_id not in existing]
    for event in new_events:
        lesson_document = LessonDocument.model_validate(lessons[event.lesson_id].payload)
        _validate_event_references(event, lesson_document)

    accepted_ids: list[str] = []
    for event in new_events:
        statement = sqlite_insert(Interaction).values(
            event_id=event.event_id,
            session_id=event.session_id,
            lesson_id=event.lesson_id,
            event_type=event.type,
            occurred_at=event.occurred_at,
            payload=event.payload,
        )
        inserted_id = db.scalar(
            statement.on_conflict_do_nothing(index_elements=["event_id"]).returning(
                Interaction.event_id
            )
        )
        if inserted_id is not None:
            accepted_ids.append(event.event_id)
        else:
            concurrent_stored = db.scalar(
                select(Interaction).where(Interaction.event_id == event.event_id)
            )
            if concurrent_stored is None or _stored_event_fingerprint(
                concurrent_stored
            ) != _input_event_fingerprint(event):
                db.rollback()
                raise EventConflictError(
                    f"event_id reused with different content: {event.event_id}"
                )
            duplicates += 1
    db.commit()

    accepted_id_set = set(accepted_ids)
    accepted_types = {event.type for event in new_events if event.event_id in accepted_id_set}
    replay_types = accepted_types or {event.type for event in unique_events.values()}
    if rebuild_derived and replay_types & PROFICIENCY_EVENT_TYPES:
        ensure_derived_states(db)
    generation_triggers = [
        event.event_id
        for event in new_events
        if event.event_id in accepted_id_set and event.type in GENERATION_TRIGGER_EVENT_TYPES
    ]
    if generation_triggers:
        ensure_generation_task(
            db,
            trigger_event_ids=generation_triggers,
            proficiency_current=rebuild_derived,
        )
    return EventRecordResult(
        accepted=len(accepted_ids),
        duplicates=duplicates,
        accepted_event_ids=accepted_ids,
    )


def _validate_event_references(event: InteractionIn, document: LessonDocument) -> None:
    if event.type == "term.revealed":
        term_key = event.payload["term_key"]
        if term_key not in document.term_catalog():
            raise ValueError(f"unknown term key for lesson: {term_key}")
    elif event.type == "translation.revealed" and event.payload["scope"] == "sentence":
        sentence_key = event.payload["sentence_key"]
        if sentence_key not in document.sentence_terms():
            raise ValueError(f"unknown sentence key for lesson: {sentence_key}")
    elif event.type == "translation.revealed" and event.payload["scope"] == "grammar":
        occurrence_key = event.payload["occurrence_key"]
        construction_key = event.payload["construction_key"]
        sentence_key = event.payload["sentence_key"]
        occurrence = document.grammar_occurrences().get(occurrence_key)
        if occurrence is None:
            raise ValueError(f"unknown grammar occurrence key for lesson: {occurrence_key}")
        expected_sentence = document.grammar_occurrence_sentences()[occurrence_key]
        if sentence_key != expected_sentence:
            raise ValueError(
                f"grammar occurrence {occurrence_key} does not belong to sentence: {sentence_key}"
            )
        if construction_key != occurrence.construction_key:
            raise ValueError(
                f"grammar occurrence {occurrence_key} does not use construction: {construction_key}"
            )


def _validate_term_keys_across_lessons(db: Session, document: LessonDocument) -> None:
    incoming = document.term_catalog()
    lessons = db.scalars(
        select(Lesson).where(Lesson.key != document.key).order_by(Lesson.imported_at, Lesson.id)
    )
    for lesson in lessons:
        existing_document = lesson_document(lesson)
        if (
            existing_document.learning_language != document.learning_language
            or existing_document.translation_language != document.translation_language
        ):
            continue
        existing = existing_document.term_catalog()
        for key in incoming.keys() & existing.keys():
            if incoming[key] != existing[key]:
                raise ValueError(f"term key has conflicting definitions across lessons: {key}")


def _reader_progress(session_id: str, events: Sequence[Interaction]) -> ReaderProgress:
    terms: set[str] = set()
    sentences: set[str] = set()
    grammar_occurrences: set[str] = set()
    full_translation = False
    started = False
    completed = False
    rating: Literal[-1, 1] | None = None
    feedback: list[FeedbackTag] = []
    for event in events:
        if event.event_type == "lesson.started":
            started = True
        elif event.event_type == "term.revealed":
            term_key = event.payload.get("term_key")
            if isinstance(term_key, str):
                terms.add(term_key)
        elif event.event_type == "translation.revealed":
            scope = event.payload.get("scope")
            if scope == "lesson":
                full_translation = True
            elif scope == "sentence" and isinstance(event.payload.get("sentence_key"), str):
                sentences.add(event.payload["sentence_key"])
            elif scope == "grammar" and isinstance(event.payload.get("occurrence_key"), str):
                grammar_occurrences.add(event.payload["occurrence_key"])
        elif event.event_type == "lesson.completed":
            completed = True
        elif event.event_type == "lesson.rated":
            if event.payload.get("rating") in {-1, 1}:
                rating = event.payload["rating"]
            feedback = feedback_from_payload(event.payload)
    return ReaderProgress(
        session_id=session_id,
        started=started,
        completed=completed,
        rating=rating,
        feedback=feedback,
        revealed_term_keys=sorted(terms),
        revealed_sentence_keys=sorted(sentences),
        revealed_grammar_occurrence_keys=sorted(grammar_occurrences),
        full_translation_revealed=full_translation,
    )


def _input_event_fingerprint(event: InteractionIn) -> str:
    return _event_fingerprint(event, event.type)


def _stored_event_fingerprint(event: Interaction) -> str:
    return _event_fingerprint(event, event.event_type)


def _event_fingerprint(event: InteractionIn | Interaction, event_type: str) -> str:
    return canonical_json(
        {
            "event_id": event.event_id,
            "session_id": event.session_id,
            "lesson_id": event.lesson_id,
            "type": event_type,
            "occurred_at": as_utc(event.occurred_at).isoformat(),
            "payload": event.payload,
        }
    )
