"""Replay reading evidence into the derived overall proficiency estimate."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, get_args

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.calibration import (
    CalibrationAttempt,
    CalibrationPrior,
    OrdinaryReadingAttempt,
    ProbeObservation,
    estimate_calibration,
)
from server.clock import as_utc, utc_now
from server.learning_units import LearningUnitIndex, build_learning_unit_index
from server.lesson_content import body_sentences, lesson_document
from server.models import Interaction, Lesson, ProficiencyState, Profile
from server.profile_activation import PROFILE_LEVEL_SOURCE_KEY, questionnaire_seed
from server.reading_evidence import finite_number, payload_is_qualified_reading
from server.schemas import FeedbackTag, LessonDocument, LevelSource, is_proper_noun_pos

ADAPTIVE_TARGET_DIFFICULTY_METADATA_KEY = "adaptive_target_difficulty"
LEGACY_LESSON_DIFFICULTY_RELIABILITY = 0.5
SELF_REPORTED_PRIOR_VARIANCE = 0.18**2


def rebuild_proficiency_state(
    db: Session,
    *,
    profile: Profile,
    source: LevelSource,
    evidence_cursor: int | None,
) -> ProficiencyState:
    """Rebuild placement plus low-weight adaptation from append-only reading evidence."""

    lessons = db.scalars(
        select(Lesson)
        .where(
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
        )
        .order_by(Lesson.imported_at, Lesson.id)
    ).all()
    all_documents = [lesson_document(lesson) for lesson in lessons]
    units = build_learning_unit_index(all_documents)
    documents = {
        lesson.id: document for lesson, document in zip(lessons, all_documents, strict=True)
    }
    grouped: dict[tuple[int, str], list[Interaction]] = defaultdict(list)
    if documents:
        interactions = db.scalars(
            select(Interaction)
            .where(Interaction.lesson_id.in_(documents))
            .order_by(Interaction.occurred_at, Interaction.id)
        ).all()
        for interaction in interactions:
            grouped[(interaction.lesson_id, interaction.session_id)].append(interaction)

    calibration_attempts: dict[int, tuple[tuple[bool, datetime, int], CalibrationAttempt]] = {}
    reading_attempts: dict[int, tuple[tuple[bool, datetime, int], OrdinaryReadingAttempt]] = {}
    lesson_keys = {lesson.id: lesson.key for lesson in lessons}
    for (lesson_id, session_id), events in sorted(grouped.items()):
        document = documents[lesson_id]
        completion = _first_completion(events)
        if completion is None:
            continue
        occurred_at = as_utc(completion.occurred_at)
        # A qualified reading outranks an earlier too-short one of the same lesson.
        rank = (not payload_is_qualified_reading(completion.payload), occurred_at, completion.id)
        if evidence_cursor is not None and completion.id <= evidence_cursor:
            continue
        if document.calibration is not None:
            revealed, translated_sentences, full_translation = _session_observations(
                events, completion
            )
            probe_sentences = document.calibration_probe_sentences()
            attempt = CalibrationAttempt(
                attempt_id=f"{lesson_keys[lesson_id]}:{session_id}",
                active_seconds=_payload_number(completion.payload, "active_seconds"),
                completion_ratio=_payload_number(completion.payload, "completion_ratio"),
                full_translation_revealed=full_translation,
                occurred_at=occurred_at,
                probes=tuple(
                    ProbeObservation(
                        term_key=probe.term_key,
                        difficulty=probe.difficulty,
                        revealed=probe.term_key in revealed,
                        sentence_translated=(
                            probe_sentences[probe.term_key] in translated_sentences
                        ),
                    )
                    for probe in document.calibration.probes
                    if units.resolve(probe.term_key) is not None
                ),
            )
            calibration_candidate = (rank, attempt)
            previous_calibration = calibration_attempts.get(lesson_id)
            if previous_calibration is None or rank < previous_calibration[0]:
                calibration_attempts[lesson_id] = calibration_candidate
            continue

        reading = _ordinary_reading_attempt(
            lesson_key=lesson_keys[lesson_id],
            session_id=session_id,
            document=document,
            units=units,
            events=events,
            completion=completion,
        )
        reading_candidate = (rank, reading)
        previous_reading = reading_attempts.get(lesson_id)
        if previous_reading is None or rank < previous_reading[0]:
            reading_attempts[lesson_id] = reading_candidate

    seed = questionnaire_seed(profile)
    if seed is None and source == "self_reported":
        seed = (profile.difficulty, SELF_REPORTED_PRIOR_VARIANCE)
    prior = CalibrationPrior(*seed) if seed is not None else None
    estimate = estimate_calibration(
        (value[1] for value in calibration_attempts.values()),
        prior=prior,
        as_of=utc_now(),
        reading_attempts=(value[1] for value in reading_attempts.values()),
    )
    state = db.get(ProficiencyState, 1) or ProficiencyState(id=1)
    state.status = estimate.status
    state.estimate = estimate.difficulty
    state.lower = estimate.lower
    state.upper = estimate.upper
    state.level = estimate.level
    state.lower_level = estimate.lower_level
    state.upper_level = estimate.upper_level
    state.qualified_attempts = estimate.qualified_attempts
    state.usable_probes = estimate.usable_probes
    state.qualified_readings = estimate.qualified_readings
    state.updated_at = utc_now()
    db.add(state)

    if source == "unknown" and estimate.status in {"rough", "stable"}:
        preferences = dict(profile.preferences)
        preferences[PROFILE_LEVEL_SOURCE_KEY] = "estimated"
        profile.preferences = preferences
    db.commit()
    db.refresh(state)
    return state


def _events_through(
    events: Sequence[Interaction],
    completion: Interaction,
) -> list[Interaction]:
    cutoff = (as_utc(completion.occurred_at), completion.id)
    return [event for event in events if (as_utc(event.occurred_at), event.id) <= cutoff]


def _session_observations(
    events: Sequence[Interaction],
    completion: Interaction,
) -> tuple[set[str], set[str], bool]:
    """Return revealed term keys, translated sentence keys, and full-translation use."""

    observed_events = _events_through(events, completion)
    revealed = {
        value
        for event in observed_events
        if event.event_type == "term.revealed"
        if isinstance((value := event.payload.get("term_key")), str)
    }
    translated_sentences = {
        value
        for event in observed_events
        if event.event_type == "translation.revealed"
        if event.payload.get("scope") == "sentence"
        if isinstance((value := event.payload.get("sentence_key")), str)
    }
    full_translation = any(
        event.event_type == "translation.revealed" and event.payload.get("scope") == "lesson"
        for event in observed_events
    )
    return revealed, translated_sentences, full_translation


def _ordinary_reading_terms(
    document: LessonDocument,
    units: LearningUnitIndex,
) -> set[str]:
    targets = {
        canonical
        for key in document.target_term_keys
        if (canonical := units.resolve(key)) is not None
    }
    eligible: set[str] = set()
    for key, term in document.term_catalog().items():
        canonical = units.resolve(key)
        if canonical is not None and canonical not in targets and not is_proper_noun_pos(term.pos):
            eligible.add(canonical)
    return eligible


def _ordinary_reading_attempt(
    *,
    lesson_key: str,
    session_id: str,
    document: LessonDocument,
    units: LearningUnitIndex,
    events: Sequence[Interaction],
    completion: Interaction,
) -> OrdinaryReadingAttempt:
    revealed, translated_sentences, full_translation = _session_observations(events, completion)
    eligible_terms = _ordinary_reading_terms(document, units)
    revealed_units = {
        canonical for key in revealed if (canonical := units.resolve(key)) is not None
    }
    sentence_keys = {sentence.key for sentence in body_sentences(document)}
    difficulty, reliability = _ordinary_lesson_difficulty(document)
    return OrdinaryReadingAttempt(
        attempt_id=f"{lesson_key}:{session_id}",
        difficulty=difficulty,
        active_seconds=_payload_number(completion.payload, "active_seconds"),
        completion_ratio=_payload_number(completion.payload, "completion_ratio"),
        eligible_term_count=len(eligible_terms),
        revealed_term_count=len(eligible_terms & revealed_units),
        sentence_count=len(sentence_keys),
        translated_sentence_count=len(sentence_keys & translated_sentences),
        full_translation_revealed=full_translation,
        difficulty_feedback=_ordinary_difficulty_feedback(events),
        difficulty_reliability=reliability,
        occurred_at=as_utc(completion.occurred_at),
    )


def _ordinary_lesson_difficulty(document: LessonDocument) -> tuple[float, float]:
    value = document.metadata.get(ADAPTIVE_TARGET_DIFFICULTY_METADATA_KEY)
    if value is None:
        return document.difficulty, LEGACY_LESSON_DIFFICULTY_RELIABILITY
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{ADAPTIVE_TARGET_DIFFICULTY_METADATA_KEY} must be between 0 and 1")
    return float(value), 1.0


def _ordinary_difficulty_feedback(
    events: Sequence[Interaction],
) -> tuple[Literal["easier", "more_challenging"], ...]:
    feedback: list[Literal["easier", "more_challenging"]] = []
    for event in events:
        if event.event_type != "lesson.rated":
            continue
        values = event.payload.get("feedback")
        if not isinstance(values, list) or any(
            not isinstance(tag, str) or tag not in get_args(FeedbackTag) for tag in values
        ):
            continue
        for tag in values:
            if tag in {"easier", "more_challenging"} and tag not in feedback:
                feedback.append(tag)
    return tuple(feedback)


def _first_completion(events: Sequence[Interaction]) -> Interaction | None:
    completions = [event for event in events if event.event_type == "lesson.completed"]
    return min(
        completions,
        key=lambda event: (
            not payload_is_qualified_reading(event.payload),
            as_utc(event.occurred_at),
            event.id,
        ),
        default=None,
    )


def _payload_number(payload: Mapping[str, Any], key: str) -> float:
    return finite_number(payload.get(key)) or 0.0
