"""Read-only aggregation for the profile-scoped Words view.

Lesson term keys remain the immutable display and interaction boundary. The shared learning-unit
resolver maps exact-definition aliases onto one canonical curriculum key and excludes conservative
language-specific productive expressions. No stemming or fuzzy lemma merging happens here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.clock import earliest_datetime, latest_datetime, optional_utc
from server.learning_units import LearningUnitIndex, build_learning_unit_index
from server.lesson_activity import (
    lesson_exposure_times,
    opened_lesson_interactions,
    profile_lessons,
    require_profile,
)
from server.lesson_content import lesson_document, lexical_runs
from server.models import LexemeState
from server.schemas import (
    LessonTerm,
    RelatedWordSense,
    WordsState,
    WordSurfaceForm,
    WordView,
)


@dataclass
class _WordAggregate:
    definition: LessonTerm
    surface_forms: Counter[str] = field(default_factory=Counter)
    lesson_ids: set[int] = field(default_factory=set)
    first_exposed_at: datetime | None = None
    last_exposed_at: datetime | None = None


def get_words_state(db: Session) -> WordsState:
    """Return terms from opened lessons without creating learning evidence.

    Any stored interaction proves that its lesson was opened, even if an older client omitted the
    ``lesson.started`` event. Content occurrences count annotated runs in each distinct opened
    lesson, not the number of browser renders. Literal clicks stay raw, while a composite click is
    attributed to its selected components. A session still counts at most once per canonical term.
    """

    profile = require_profile(db)
    lessons = profile_lessons(db, profile)
    documents = {lesson.id: lesson_document(lesson) for lesson in lessons}
    unit_index = build_learning_unit_index(documents.values())
    opened_lessons, interactions = opened_lesson_interactions(db, lessons)
    if not opened_lessons:
        return WordsState(
            learning_language=profile.learning_language,
            translation_language=profile.translation_language,
            words=[],
        )

    exposure_times = lesson_exposure_times(interactions)
    states = {
        state.term_key: state
        for state in db.scalars(
            select(LexemeState).where(
                LexemeState.learning_language == profile.learning_language,
                LexemeState.translation_language == profile.translation_language,
                LexemeState.term_key.in_(unit_index.definitions),
            )
        )
    }
    missing_states = unit_index.definitions.keys() - states.keys()
    if missing_states:
        raise RuntimeError(f"lexeme state is missing terms: {sorted(missing_states)}")
    establishment = {
        key: state.qualified_exposures + state.reveal_failures for key, state in states.items()
    }

    aggregates: dict[str, _WordAggregate] = {}
    for lesson in opened_lessons:
        opened_at = exposure_times[lesson.id]
        for run in lexical_runs(documents[lesson.id]):
            term = run.term
            if term is None:
                raise RuntimeError("lexical run is missing its term")
            for term_key, projected_surface in _projected_learning_units(
                unit_index, term, run.text, establishment
            ):
                definition = _canonical_definition(unit_index, term_key)
                aggregate = aggregates.get(term_key)
                if aggregate is None:
                    aggregate = _WordAggregate(definition=definition)
                    aggregates[term_key] = aggregate
                aggregate.surface_forms[projected_surface] += 1
                aggregate.lesson_ids.add(lesson.id)
                first, last = opened_at
                aggregate.first_exposed_at = earliest_datetime(
                    aggregate.first_exposed_at,
                    first,
                )
                aggregate.last_exposed_at = latest_datetime(aggregate.last_exposed_at, last)

    raw_reveals: Counter[str] = Counter()
    reveal_sessions: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for interaction in interactions:
        if interaction.event_type != "term.revealed":
            continue
        display_key = interaction.payload.get("term_key")
        if not isinstance(display_key, str):
            continue
        canonical = unit_index.resolve(display_key)
        if canonical is not None:
            if canonical not in aggregates:
                continue
            raw_reveals[canonical] += 1
            reveal_sessions[canonical].add((interaction.lesson_id, interaction.session_id))
            continue
        selected = unit_index.select_evidence_targets(display_key, establishment)
        if len(set(selected)) < 2:
            continue
        for term_key in selected:
            if term_key in aggregates:
                reveal_sessions[term_key].add((interaction.lesson_id, interaction.session_id))

    related_senses = _related_senses(aggregates, states)
    words = [
        _word_view(
            term_key,
            aggregate,
            states[term_key],
            related_senses=related_senses[term_key],
            raw_reveal_count=raw_reveals[term_key],
            counted_reveal_sessions=len(reveal_sessions[term_key]),
        )
        for term_key, aggregate in aggregates.items()
    ]
    words.sort(
        key=lambda word: (
            word.frequency_rank is None,
            word.frequency_rank or 0,
            word.lemma.casefold(),
            word.pos.casefold(),
            word.key,
        )
    )
    return WordsState(
        learning_language=profile.learning_language,
        translation_language=profile.translation_language,
        words=words,
    )


def _canonical_definition(index: LearningUnitIndex, term_key: str) -> LessonTerm:
    try:
        return index.definitions[term_key]
    except KeyError as error:
        raise RuntimeError(f"learning-unit definition is missing: {term_key}") from error


def _projected_learning_units(
    index: LearningUnitIndex,
    term: LessonTerm,
    surface: str,
    establishment: dict[str, float],
) -> list[tuple[str, str]]:
    canonical = index.resolve(term.key)
    if canonical is not None:
        return [(canonical, surface)]
    candidates = index.component_candidates(term.key)
    selected = index.select_evidence_targets(term.key, establishment)
    if len(selected) != len(candidates) or len(set(selected)) < 2:
        return []
    return [
        (term_key, item.component.surface)
        for term_key, item in zip(selected, candidates, strict=True)
    ]


def _word_view(
    term_key: str,
    aggregate: _WordAggregate,
    state: LexemeState,
    *,
    related_senses: list[RelatedWordSense],
    raw_reveal_count: int,
    counted_reveal_sessions: int,
) -> WordView:
    first_exposed_at = aggregate.first_exposed_at
    last_exposed_at = aggregate.last_exposed_at
    if first_exposed_at is None or last_exposed_at is None:
        raise RuntimeError(f"word has no exposure time: {term_key}")
    return WordView(
        key=term_key,
        lemma=state.lemma,
        pos=state.pos,
        gloss=state.gloss,
        pronunciation=state.pronunciation,
        frequency_rank=state.frequency_rank,
        surface_forms=[
            WordSurfaceForm(text=surface, occurrences=count)
            for surface, count in sorted(
                aggregate.surface_forms.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        related_senses=related_senses,
        occurrence_count=sum(aggregate.surface_forms.values()),
        exposed_lesson_count=len(aggregate.lesson_ids),
        raw_reveal_count=raw_reveal_count,
        counted_reveal_sessions=counted_reveal_sessions,
        qualified_exposures=state.qualified_exposures,
        mastery=state.mastery,
        stability_days=state.stability_days,
        reveal_failures=state.reveal_failures,
        first_exposed_at=first_exposed_at,
        last_exposed_at=last_exposed_at,
        last_revealed_at=optional_utc(state.last_revealed_at),
        next_due_at=optional_utc(state.next_due_at),
    )


def _related_senses(
    aggregates: dict[str, _WordAggregate], states: dict[str, LexemeState]
) -> dict[str, list[RelatedWordSense]]:
    """Link exact-lemma alternatives without combining their learning evidence."""

    keys_by_lemma: dict[str, list[str]] = defaultdict(list)
    for key, aggregate in aggregates.items():
        keys_by_lemma[aggregate.definition.lemma].append(key)

    return {
        key: [
            RelatedWordSense(
                key=related_key,
                pos=states[related_key].pos,
                gloss=states[related_key].gloss,
                pronunciation=states[related_key].pronunciation,
            )
            for related_key in sorted(
                keys_by_lemma[aggregate.definition.lemma],
                key=lambda related_key: (
                    states[related_key].pos.casefold(),
                    states[related_key].gloss.casefold(),
                    related_key,
                ),
            )
            if related_key != key
        ]
        for key, aggregate in aggregates.items()
    }
