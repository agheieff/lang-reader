"""Replay indirect reading evidence into derived Han-character recognition state."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from server.clock import as_utc, utc_now
from server.han import character_tracking_available, han_characters
from server.learning_units import LearningUnitIndex, learning_unit_indexes
from server.lesson_content import all_sentences, lesson_document
from server.memory_model import (
    AGAIN,
    GOOD,
    MemoryPolicy,
    MemoryState,
    interval_days,
    knowledge,
    review,
)
from server.models import CharacterState, Interaction, Lesson
from server.reading_evidence import (
    DEFAULT_READING_EVIDENCE_POLICY,
    ReadingEvidencePolicy,
    qualified_completion_and_considered,
)
from server.schemas import LessonDocument

# Characters get their own memory policy; reveals are inferred from word clicks, so they are
# applied as partial lapses whose weight is the configured failure mass.
CHARACTER_MEMORY_POLICY = MemoryPolicy(passive_confidence=0.45)
CHARACTER_PRIOR = 0.5


@dataclass(frozen=True)
class CharacterLearningPolicy:
    """Tunable weights for evidence inferred from ordinary word-based reading."""

    qualification: ReadingEvidencePolicy = DEFAULT_READING_EVIDENCE_POLICY
    new_context_success_bonus: float = 0.25
    single_character_failure_mass: float = 0.4
    multi_character_failure_mass: float = 0.2
    failure_cap_per_session: float = 0.5
    memory: MemoryPolicy = CHARACTER_MEMORY_POLICY

    def __post_init__(self) -> None:
        values = (
            self.new_context_success_bonus,
            self.single_character_failure_mass,
            self.multi_character_failure_mass,
            self.failure_cap_per_session,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("character learning weights must be finite and non-negative")
        if (
            max(
                self.single_character_failure_mass,
                self.multi_character_failure_mass,
            )
            > self.failure_cap_per_session
        ):
            raise ValueError("character failure mass must not exceed the per-session cap")
        if self.failure_cap_per_session > 1:
            raise ValueError("the per-session character failure cap must be at most one lapse")


DEFAULT_CHARACTER_LEARNING_POLICY = CharacterLearningPolicy()


@dataclass
class _CharacterEstimate:
    character: str
    learning_language: str
    translation_language: str
    memory: MemoryState | None = None
    alpha: float = 2.0
    beta: float = 2.0
    qualified_exposures: int = 0
    inferred_failure_mass: float = 0.0
    direct_successes: int = 0
    direct_failures: int = 0
    lesson_ids: set[int] = field(default_factory=set)
    word_context_keys: set[str] = field(default_factory=set)
    failure_sessions: set[tuple[int, str]] = field(default_factory=set)
    first_evidence_at: datetime | None = None
    last_evidence_at: datetime | None = None
    last_inferred_failure_at: datetime | None = None
    next_due_at: datetime | None = None

    def knowledge_at(self, at: datetime, policy: CharacterLearningPolicy) -> float:
        return knowledge(self.memory, prior=CHARACTER_PRIOR, at=at, policy=policy.memory)


@dataclass(frozen=True)
class _RunUnit:
    sentence_key: str
    term_key: str | None
    characters: tuple[str, ...]
    context_key: str | None

    @property
    def distinct_characters(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.characters))


@dataclass(frozen=True)
class _CharacterEvidence:
    at: datetime
    order: int
    kind: Literal["failure", "success"]
    learning_language: str
    translation_language: str
    lesson_id: int
    session_id: str
    characters: tuple[str, ...]
    span_length: int
    contexts: Mapping[str, frozenset[str]]


def rebuild_character_states(
    db: Session,
    *,
    policy: CharacterLearningPolicy = DEFAULT_CHARACTER_LEARNING_POLICY,
) -> list[CharacterState]:
    """Replace the disposable character cache with a deterministic evidence replay."""

    lessons = db.scalars(select(Lesson).order_by(Lesson.imported_at, Lesson.id)).all()
    documents = {lesson.id: lesson_document(lesson) for lesson in lessons}
    interactions = db.scalars(
        select(Interaction).order_by(Interaction.occurred_at, Interaction.id)
    ).all()
    estimates = _replay_character_states(documents, interactions, policy)

    db.execute(delete(CharacterState))
    rows = [_state_row(state) for _key, state in sorted(estimates.items())]
    db.add_all(rows)
    db.commit()
    return rows


def _replay_character_states(
    documents: Mapping[int, LessonDocument],
    interactions: Sequence[Interaction],
    policy: CharacterLearningPolicy,
) -> dict[tuple[str, str, str], _CharacterEstimate]:
    indexes = learning_unit_indexes(
        document
        for document in documents.values()
        if character_tracking_available(document.learning_language)
    )
    units = {
        lesson_id: _run_units(
            document,
            indexes[(document.learning_language, document.translation_language)],
        )
        for lesson_id, document in documents.items()
        if character_tracking_available(document.learning_language)
    }
    estimates = _initial_estimates(documents, units)
    evidence = _ordered_evidence(documents, units, interactions, policy)

    failure_totals: dict[tuple[int, str, str], float] = defaultdict(float)
    first_reading_session: dict[int, str] = {}
    for item in evidence:
        if item.kind == "success":
            reread = first_reading_session.setdefault(item.lesson_id, item.session_id) != (
                item.session_id
            )
            for character in item.characters:
                state = estimates.get((*_languages(item), character))
                if state is not None:
                    _apply_success(state, item, policy, reread=reread)
            continue

        weights = _failure_weights(item, estimates, policy)
        for character, requested_weight in weights.items():
            state = estimates.get((*_languages(item), character))
            if state is None:
                continue
            cap_key = (item.lesson_id, item.session_id, character)
            weight = min(
                requested_weight,
                policy.failure_cap_per_session - failure_totals[cap_key],
            )
            if weight <= 1e-12:
                continue
            _apply_failure(state, item, weight, policy)
            failure_totals[cap_key] = round(failure_totals[cap_key] + weight, 12)
    return estimates


def _run_units(document: LessonDocument, index: LearningUnitIndex) -> tuple[_RunUnit, ...]:
    units: list[_RunUnit] = []
    for sentence in all_sentences(document):
        for run in sentence.runs:
            characters = han_characters(run.text)
            if not characters:
                continue
            units.append(
                _RunUnit(
                    sentence_key=sentence.key,
                    term_key=run.term.key if run.term is not None else None,
                    characters=characters,
                    context_key=index.resolve(run.term.key) if run.term is not None else None,
                )
            )
    return tuple(units)


def _initial_estimates(
    documents: Mapping[int, LessonDocument],
    units: Mapping[int, tuple[_RunUnit, ...]],
) -> dict[tuple[str, str, str], _CharacterEstimate]:
    estimates: dict[tuple[str, str, str], _CharacterEstimate] = {}
    for lesson_id, lesson_units in units.items():
        document = documents[lesson_id]
        for unit in lesson_units:
            for character in unit.distinct_characters:
                key = (document.learning_language, document.translation_language, character)
                estimates.setdefault(
                    key,
                    _CharacterEstimate(
                        character=character,
                        learning_language=document.learning_language,
                        translation_language=document.translation_language,
                    ),
                )
    return estimates


def _ordered_evidence(
    documents: Mapping[int, LessonDocument],
    units: Mapping[int, tuple[_RunUnit, ...]],
    interactions: Sequence[Interaction],
    policy: CharacterLearningPolicy,
) -> list[_CharacterEvidence]:
    sessions: dict[tuple[int, str], list[Interaction]] = defaultdict(list)
    for interaction in interactions:
        if interaction.lesson_id in units:
            sessions[(interaction.lesson_id, interaction.session_id)].append(interaction)

    evidence: list[_CharacterEvidence] = []
    for (lesson_id, _session_id), events in sessions.items():
        evidence.extend(
            _session_evidence(
                lesson_id,
                documents[lesson_id],
                units[lesson_id],
                events,
                policy,
            )
        )
    evidence.sort(
        key=lambda item: (
            item.at,
            item.order,
            0 if item.kind == "failure" else 1,
            item.characters,
        )
    )
    return evidence


def _session_evidence(
    lesson_id: int,
    document: LessonDocument,
    units: tuple[_RunUnit, ...],
    events: Sequence[Interaction],
    policy: CharacterLearningPolicy,
) -> list[_CharacterEvidence]:
    qualified_completion, considered = qualified_completion_and_considered(
        events, policy.qualification
    )
    reveals: dict[tuple[str, str | None], Interaction] = {}
    translated_sentences: set[str] = set()
    full_translation = False
    for event in considered:
        if event.event_type == "term.revealed":
            term_key = event.payload.get("term_key")
            if isinstance(term_key, str):
                sentence_key = event.payload.get("sentence_key")
                reveal_key = (
                    term_key,
                    sentence_key if isinstance(sentence_key, str) else None,
                )
                reveals.setdefault(reveal_key, event)
        elif event.event_type == "translation.revealed":
            scope = event.payload.get("scope")
            if scope == "lesson":
                full_translation = True
            elif scope == "sentence" and isinstance(event.payload.get("sentence_key"), str):
                translated_sentences.add(event.payload["sentence_key"])

    result: list[_CharacterEvidence] = []
    excluded_characters: set[str] = set()
    for (term_key, sentence_key), event in reveals.items():
        candidates = _matching_units(units, term_key, sentence_key)
        excluded_characters.update(
            character for unit in candidates for character in unit.distinct_characters
        )
        resolved = _unambiguous_reveal(candidates)
        if resolved is None:
            continue
        contexts = _contexts_by_character(candidates, resolved.distinct_characters)
        result.append(
            _evidence(
                "failure",
                lesson_id,
                document,
                event,
                resolved.distinct_characters,
                span_length=len(resolved.characters),
                contexts=contexts,
            )
        )

    for unit in units:
        if unit.sentence_key in translated_sentences:
            excluded_characters.update(unit.distinct_characters)
    if qualified_completion is None or full_translation:
        return result

    clean_units = tuple(
        unit
        for unit in units
        if unit.sentence_key not in translated_sentences and not _unit_was_revealed(unit, reveals)
    )
    clean_characters = tuple(
        sorted(
            {
                character
                for unit in clean_units
                for character in unit.distinct_characters
                if character not in excluded_characters
            }
        )
    )
    if clean_characters:
        result.append(
            _evidence(
                "success",
                lesson_id,
                document,
                qualified_completion,
                clean_characters,
                span_length=len(clean_characters),
                contexts=_contexts_by_character(clean_units, clean_characters),
            )
        )
    return result


def _matching_units(
    units: tuple[_RunUnit, ...],
    term_key: str,
    sentence_key: str | None,
) -> tuple[_RunUnit, ...]:
    return tuple(
        unit
        for unit in units
        if unit.term_key == term_key and (sentence_key is None or unit.sentence_key == sentence_key)
    )


def _unambiguous_reveal(candidates: tuple[_RunUnit, ...]) -> _RunUnit | None:
    sequences = {candidate.characters for candidate in candidates}
    return candidates[0] if candidates and len(sequences) == 1 else None


def _unit_was_revealed(
    unit: _RunUnit,
    reveals: Mapping[tuple[str, str | None], Interaction],
) -> bool:
    if unit.term_key is None:
        return False
    return (unit.term_key, unit.sentence_key) in reveals or (unit.term_key, None) in reveals


def _contexts_by_character(
    units: Sequence[_RunUnit],
    characters: Sequence[str],
) -> dict[str, frozenset[str]]:
    selected = set(characters)
    contexts: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        if unit.context_key is None:
            continue
        for character in unit.distinct_characters:
            if character in selected:
                contexts[character].add(unit.context_key)
    return {character: frozenset(contexts[character]) for character in characters}


def _evidence(
    kind: Literal["failure", "success"],
    lesson_id: int,
    document: LessonDocument,
    event: Interaction,
    characters: tuple[str, ...],
    *,
    span_length: int,
    contexts: Mapping[str, frozenset[str]],
) -> _CharacterEvidence:
    return _CharacterEvidence(
        at=as_utc(event.occurred_at),
        order=event.id,
        kind=kind,
        learning_language=document.learning_language,
        translation_language=document.translation_language,
        lesson_id=lesson_id,
        session_id=event.session_id,
        characters=characters,
        span_length=span_length,
        contexts=contexts,
    )


def _languages(evidence: _CharacterEvidence) -> tuple[str, str]:
    return evidence.learning_language, evidence.translation_language


def _failure_weights(
    evidence: _CharacterEvidence,
    estimates: Mapping[tuple[str, str, str], _CharacterEstimate],
    policy: CharacterLearningPolicy,
) -> dict[str, float]:
    total_mass = (
        policy.single_character_failure_mass
        if evidence.span_length == 1
        else policy.multi_character_failure_mass
    )
    selected = [
        estimates[(*_languages(evidence), character)]
        for character in evidence.characters
        if (*_languages(evidence), character) in estimates
    ]
    scores = [
        max(
            1e-9,
            (1.0 - state.knowledge_at(evidence.at, policy))
            * (1.0 + min(1.0, 4.0 / (state.alpha + state.beta))),
        )
        for state in selected
    ]
    total_score = sum(scores)
    return {
        state.character: total_mass * score / total_score
        for state, score in zip(selected, scores, strict=True)
    }


def _apply_failure(
    state: _CharacterEstimate,
    evidence: _CharacterEvidence,
    weight: float,
    policy: CharacterLearningPolicy,
) -> None:
    state.memory = review(state.memory, AGAIN, evidence.at, weight=weight, policy=policy.memory)
    state.beta += weight
    state.inferred_failure_mass = round(state.inferred_failure_mass + weight, 12)
    state.failure_sessions.add((evidence.lesson_id, evidence.session_id))
    _record_evidence_context(state, evidence)
    state.last_inferred_failure_at = evidence.at
    state.next_due_at = evidence.at + timedelta(days=interval_days(state.memory.stability_days))


def _apply_success(
    state: _CharacterEstimate,
    evidence: _CharacterEvidence,
    policy: CharacterLearningPolicy,
    *,
    reread: bool = False,
) -> None:
    contexts = evidence.contexts.get(state.character, frozenset())
    has_new_context = bool(contexts - state.word_context_keys)
    weight = policy.memory.passive_confidence
    if has_new_context:
        weight *= 1.0 + policy.new_context_success_bonus
    if reread:
        weight *= policy.memory.reread_weight
    weight = min(1.0, weight)
    state.memory = review(state.memory, GOOD, evidence.at, weight=weight, policy=policy.memory)
    state.alpha += weight
    if not reread:
        state.qualified_exposures += 1
    _record_evidence_context(state, evidence)
    state.next_due_at = evidence.at + timedelta(days=interval_days(state.memory.stability_days))


def _record_evidence_context(
    state: _CharacterEstimate,
    evidence: _CharacterEvidence,
) -> None:
    state.lesson_ids.add(evidence.lesson_id)
    state.word_context_keys.update(evidence.contexts.get(state.character, frozenset()))
    state.first_evidence_at = state.first_evidence_at or evidence.at
    state.last_evidence_at = evidence.at


def _state_row(state: _CharacterEstimate) -> CharacterState:
    return CharacterState(
        learning_language=state.learning_language,
        translation_language=state.translation_language,
        character=state.character,
        alpha=state.alpha,
        beta=state.beta,
        stability_days=state.memory.stability_days if state.memory is not None else 0.0,
        memory_difficulty=state.memory.difficulty if state.memory is not None else None,
        qualified_exposures=state.qualified_exposures,
        inferred_failure_sessions=len(state.failure_sessions),
        inferred_failure_mass=state.inferred_failure_mass,
        direct_successes=state.direct_successes,
        direct_failures=state.direct_failures,
        distinct_lessons=len(state.lesson_ids),
        distinct_word_contexts=len(state.word_context_keys),
        first_evidence_at=state.first_evidence_at,
        last_evidence_at=state.last_evidence_at,
        last_inferred_failure_at=state.last_inferred_failure_at,
        next_due_at=state.next_due_at,
        updated_at=utc_now(),
    )
