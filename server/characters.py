"""Opened-text observations plus replayable Han-character recognition estimates."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.character_word_support import character_retrievability
from server.clock import earliest_datetime, latest_datetime, optional_utc, utc_now
from server.han import character_tracking_available, han_characters
from server.lesson_activity import (
    lesson_exposure_times,
    opened_lesson_interactions,
    profile_lessons,
    require_profile,
)
from server.lesson_content import all_sentences, lesson_document
from server.models import CharacterState, Profile
from server.schemas import CharactersState, CharacterView


@dataclass
class _CharacterAggregate:
    occurrence_count: int = 0
    lesson_ids: set[int] = field(default_factory=set)
    first_exposed_at: datetime | None = None
    last_exposed_at: datetime | None = None


@dataclass(frozen=True)
class _TermOccurrence:
    sentence_key: str
    characters: frozenset[str]


def get_characters_state(db: Session) -> CharactersState:
    """Project literal exposure and derived recognition from immutable history.

    No character-specific interaction is forged. A term reveal is attributed to the Han characters
    in the clicked display run. Modern events identify its sentence. For legacy events without that
    boundary, attribution is retained only when every occurrence has the same character identity.
    """

    profile = require_profile(db)
    if not character_tracking_available(profile.learning_language):
        return _empty_state(profile)

    opened_lessons, interactions = opened_lesson_interactions(db, profile_lessons(db, profile))
    if not opened_lessons:
        return _empty_state(profile)

    exposure_times = lesson_exposure_times(interactions)
    states = {
        state.character: state
        for state in db.scalars(
            select(CharacterState).where(
                CharacterState.learning_language == profile.learning_language,
                CharacterState.translation_language == profile.translation_language,
            )
        )
    }
    aggregates: dict[str, _CharacterAggregate] = {}
    term_occurrences: dict[tuple[int, str], list[_TermOccurrence]] = defaultdict(list)

    for lesson in opened_lessons:
        document = lesson_document(lesson)
        first_exposed_at, last_exposed_at = exposure_times[lesson.id]
        for sentence in all_sentences(document):
            for run in sentence.runs:
                run_characters = han_characters(run.text)
                for character, count in Counter(run_characters).items():
                    aggregate = aggregates.setdefault(character, _CharacterAggregate())
                    aggregate.occurrence_count += count
                    aggregate.lesson_ids.add(lesson.id)
                    aggregate.first_exposed_at = earliest_datetime(
                        aggregate.first_exposed_at,
                        first_exposed_at,
                    )
                    aggregate.last_exposed_at = latest_datetime(
                        aggregate.last_exposed_at,
                        last_exposed_at,
                    )
                if run.term is not None:
                    term_occurrences[(lesson.id, run.term.key)].append(
                        _TermOccurrence(sentence.key, frozenset(run_characters))
                    )

    missing_states = aggregates.keys() - states.keys()
    if missing_states:
        raise RuntimeError(f"character state is missing characters: {sorted(missing_states)}")

    raw_reveals: Counter[str] = Counter()
    reveal_sessions: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for interaction in interactions:
        if interaction.event_type != "term.revealed":
            continue
        term_key = interaction.payload.get("term_key")
        if not isinstance(term_key, str):
            continue
        sentence_key = interaction.payload.get("sentence_key")
        revealed_characters = _revealed_characters(
            term_occurrences.get((interaction.lesson_id, term_key), ()),
            sentence_key if isinstance(sentence_key, str) else None,
        )
        for character in revealed_characters:
            if character not in aggregates:
                continue
            raw_reveals[character] += 1
            reveal_sessions[character].add((interaction.lesson_id, interaction.session_id))

    character_views = [
        _character_view(
            character,
            aggregate,
            state=states[character],
            raw_reveal_count=raw_reveals[character],
            counted_reveal_sessions=len(reveal_sessions[character]),
        )
        for character, aggregate in aggregates.items()
    ]
    character_views.sort(key=lambda item: (-item.occurrence_count, item.character))
    return CharactersState(
        learning_language=profile.learning_language,
        translation_language=profile.translation_language,
        characters=character_views,
    )


def _revealed_characters(
    occurrences: tuple[_TermOccurrence, ...] | list[_TermOccurrence],
    sentence_key: str | None,
) -> frozenset[str]:
    if sentence_key is not None:
        sentence_occurrences = [
            occurrence.characters
            for occurrence in occurrences
            if occurrence.sentence_key == sentence_key
        ]
        if not sentence_occurrences:
            return frozenset()
        # A legacy sentence can repeat one key with different surfaces but does not identify the
        # exact clicked run. Attribute only characters that every possible occurrence contains.
        first, *remaining = sentence_occurrences
        return first.intersection(*remaining)

    possible = {occurrence.characters for occurrence in occurrences}
    if len(possible) != 1:
        return frozenset()
    return next(iter(possible))


def _character_view(
    character: str,
    aggregate: _CharacterAggregate,
    *,
    state: CharacterState,
    raw_reveal_count: int,
    counted_reveal_sessions: int,
) -> CharacterView:
    first_exposed_at = aggregate.first_exposed_at
    last_exposed_at = aggregate.last_exposed_at
    if first_exposed_at is None or last_exposed_at is None:
        raise RuntimeError(f"character has no exposure time: {character}")
    return CharacterView(
        character=character,
        occurrence_count=aggregate.occurrence_count,
        exposed_lesson_count=len(aggregate.lesson_ids),
        raw_reveal_count=raw_reveal_count,
        counted_reveal_sessions=counted_reveal_sessions,
        distinct_word_contexts=state.distinct_word_contexts,
        qualified_exposures=state.qualified_exposures,
        inferred_failure_sessions=state.inferred_failure_sessions,
        inferred_failure_mass=state.inferred_failure_mass,
        direct_successes=state.direct_successes,
        direct_failures=state.direct_failures,
        mastery=state.mastery,
        mastery_uncertainty=_beta_uncertainty(state.alpha, state.beta),
        retrievability=character_retrievability(state, at=utc_now()),
        stability_days=state.stability_days,
        first_exposed_at=first_exposed_at,
        last_exposed_at=last_exposed_at,
        last_evidence_at=optional_utc(state.last_evidence_at),
        next_due_at=optional_utc(state.next_due_at),
    )


def _beta_uncertainty(alpha: float, beta: float) -> float:
    total = alpha + beta
    if total <= 0:
        return 0.5
    return math.sqrt(alpha * beta / (total * total * (total + 1.0)))


def _empty_state(profile: Profile) -> CharactersState:
    return CharactersState(
        learning_language=profile.learning_language,
        translation_language=profile.translation_language,
        characters=[],
    )
