"""Deterministic grammar evidence reduction and read-side aggregation.

Lesson grammar annotations describe what was prepared; interactions remain the immutable learning
record. ``grammar_states`` is only a cache of replayed evidence and can always be discarded and
rebuilt from those two sources.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, cast

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from server.clock import as_utc, earliest_datetime, latest_datetime, optional_utc, utc_now
from server.grammar_catalog import GrammarConstruction, grammar_construction
from server.lesson_activity import (
    lesson_exposure_times,
    opened_lesson_interactions,
    profile_lessons,
    require_profile,
)
from server.lesson_content import all_sentences, lesson_document
from server.models import GrammarState as GrammarStateRow
from server.models import Interaction, Lesson, Profile
from server.reading_evidence import (
    DEFAULT_READING_EVIDENCE_POLICY,
    ReadingEvidencePolicy,
    payload_is_qualified_reading,
)
from server.schemas import (
    GrammarExampleView,
    GrammarOccurrence,
    GrammarState,
    GrammarView,
    LessonDocument,
)
from server.spaced_repetition import (
    DEFAULT_STABILITY_POLICY,
    StabilityPolicy,
    estimated_recall,
    stability_after_failure,
    stability_after_weighted_success,
)


@dataclass(frozen=True)
class GrammarLearningPolicy:
    """Tunable evidence and scheduling weights for grammar constructions."""

    qualification: ReadingEvidencePolicy = DEFAULT_READING_EVIDENCE_POLICY
    clean_success_weight: float = 1.0
    new_lesson_success_bonus: float = 0.25
    explicit_help_weight: float = 1.0
    inferred_difficulty_weight: float = 0.35
    max_inferred_vocabulary_reveals: int = 1
    explicit_help_due_days: float = 0.25
    inferred_difficulty_due_days: float = 1.5
    stability_gain_days_per_success_mass: float = 0.75
    stability: StabilityPolicy = DEFAULT_STABILITY_POLICY

    def __post_init__(self) -> None:
        values = (
            self.clean_success_weight,
            self.new_lesson_success_bonus,
            self.explicit_help_weight,
            self.inferred_difficulty_weight,
            self.explicit_help_due_days,
            self.inferred_difficulty_due_days,
            self.stability_gain_days_per_success_mass,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("grammar learning values must be finite and non-negative")
        if (
            isinstance(self.max_inferred_vocabulary_reveals, bool)
            or not isinstance(self.max_inferred_vocabulary_reveals, int)
            or self.max_inferred_vocabulary_reveals < 0
        ):
            raise ValueError("maximum inferred vocabulary reveals must be a non-negative integer")


DEFAULT_GRAMMAR_LEARNING_POLICY = GrammarLearningPolicy()

# Keep this deliberately aligned with the user-facing ``comfortable`` grammar status. A marker is
# only visual scaffolding, so hiding it requires broad, stable, mostly unassisted evidence.
COMFORTABLE_MIN_QUALIFIED_EXPOSURES = 4
COMFORTABLE_MIN_EXPOSED_LESSONS = 3
COMFORTABLE_MIN_MASTERY = 0.8
COMFORTABLE_MIN_STABILITY_DAYS = 7.0
COMFORTABLE_MAX_DIFFICULTY_SIGNAL_SHARE = 0.25

_FailureKind = Literal["explicit_help", "inferred_difficulty"]
_EvidenceKind = Literal["explicit_help", "inferred_difficulty", "success"]


@dataclass(frozen=True)
class _Definition:
    key: str
    label: str
    description: str
    category: str
    difficulty: float


@dataclass(frozen=True)
class _Occurrence:
    key: str
    construction_key: str
    run_start: int
    run_end: int
    note: str | None


@dataclass(frozen=True)
class _Sentence:
    key: str
    text: str
    translation: str | None
    term_keys: frozenset[str]
    occurrences: tuple[_Occurrence, ...]


@dataclass(frozen=True)
class _DocumentGrammar:
    definitions: Mapping[str, _Definition]
    sentences: Mapping[str, _Sentence]

    @property
    def construction_keys(self) -> frozenset[str]:
        return frozenset(
            occurrence.construction_key
            for sentence in self.sentences.values()
            for occurrence in sentence.occurrences
        )


@dataclass
class _State:
    definition: _Definition
    learning_language: str
    translation_language: str
    alpha: float = 2.0
    beta: float = 2.0
    stability: float = 0.5
    qualified_exposures: int = 0
    explicit_help_failures: int = 0
    inferred_difficulty_signals: int = 0
    lesson_ids: set[int] = field(default_factory=set)
    successful_lesson_ids: set[int] = field(default_factory=set)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    last_helped: datetime | None = None
    next_due: datetime | None = None

    @property
    def mastery(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass(frozen=True)
class _Evidence:
    at: datetime
    kind: _EvidenceKind
    learning_language: str
    translation_language: str
    construction_key: str
    lesson_id: int


@dataclass(frozen=True)
class _ExampleCandidate:
    exposed_at: datetime
    lesson_id: int
    title: str
    sentence_key: str
    text: str
    translation: str | None
    occurrence_key: str
    note: str | None


@dataclass
class _GrammarAggregate:
    definition: _Definition
    occurrence_count: int = 0
    lesson_ids: set[int] = field(default_factory=set)
    first_exposed_at: datetime | None = None
    last_exposed_at: datetime | None = None
    examples: list[_ExampleCandidate] = field(default_factory=list)


def comfortable_grammar_construction_keys(
    db: Session,
    *,
    learning_language: str,
    translation_language: str,
    construction_keys: set[str] | frozenset[str] | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Return profile-local constructions whose reader markers can be hidden.

    Authored occurrences remain untouched. This is only a read-side projection over replayed state
    and lessons that have real interactions; previewing a lesson therefore cannot advance the
    exposed-lesson requirement.
    """

    statement = select(GrammarStateRow).where(
        GrammarStateRow.learning_language == learning_language,
        GrammarStateRow.translation_language == translation_language,
    )
    if construction_keys is not None:
        if not construction_keys:
            return []
        statement = statement.where(GrammarStateRow.construction_key.in_(construction_keys))
    states = db.scalars(statement).all()
    if not states:
        return []

    exposed_lesson_counts = _opened_construction_lesson_counts(
        db,
        learning_language=learning_language,
        translation_language=translation_language,
        construction_keys={state.construction_key for state in states},
    )
    as_of = as_utc(now or utc_now())
    return sorted(
        state.construction_key
        for state in states
        if grammar_marker_can_be_hidden(
            qualified_exposures=state.qualified_exposures,
            exposed_lesson_count=exposed_lesson_counts[state.construction_key],
            mastery=state.mastery,
            stability_days=state.stability_days,
            counted_help_sessions=state.explicit_help_failures,
            inferred_difficulty_signals=state.inferred_difficulty_signals,
            next_due_at=state.next_due_at,
            now=as_of,
        )
    )


def grammar_marker_can_be_hidden(
    *,
    qualified_exposures: int,
    exposed_lesson_count: int,
    mastery: float,
    stability_days: float,
    counted_help_sessions: int,
    inferred_difficulty_signals: int,
    next_due_at: datetime | None,
    now: datetime | None = None,
) -> bool:
    """Apply the single conservative, time-aware marker-suppression policy."""

    difficulty_signals = counted_help_sessions + inferred_difficulty_signals
    evidence = qualified_exposures + difficulty_signals
    signal_share = difficulty_signals / evidence if evidence else 0.0
    return (
        qualified_exposures >= COMFORTABLE_MIN_QUALIFIED_EXPOSURES
        and exposed_lesson_count >= COMFORTABLE_MIN_EXPOSED_LESSONS
        and mastery >= COMFORTABLE_MIN_MASTERY
        and stability_days >= COMFORTABLE_MIN_STABILITY_DAYS
        and signal_share <= COMFORTABLE_MAX_DIFFICULTY_SIGNAL_SHARE
        and next_due_at is not None
        and as_utc(next_due_at) > as_utc(now or utc_now())
    )


def _opened_construction_lesson_counts(
    db: Session,
    *,
    learning_language: str,
    translation_language: str,
    construction_keys: set[str],
) -> dict[str, int]:
    counts = {construction_key: 0 for construction_key in construction_keys}
    opened_lesson_ids = set(
        db.scalars(
            select(Interaction.lesson_id)
            .join(Lesson, Interaction.lesson_id == Lesson.id)
            .where(
                Lesson.learning_language == learning_language,
                Lesson.translation_language == translation_language,
            )
            .distinct()
        )
    )
    if not opened_lesson_ids:
        return counts

    payloads = db.scalars(select(Lesson.payload).where(Lesson.id.in_(opened_lesson_ids))).all()
    for payload in payloads:
        document_keys = LessonDocument.model_validate(payload).grammar_construction_keys()
        for construction_key in construction_keys & document_keys:
            counts[construction_key] += 1
    return counts


def rebuild_grammar_states(
    db: Session,
    *,
    policy: GrammarLearningPolicy = DEFAULT_GRAMMAR_LEARNING_POLICY,
) -> list[GrammarStateRow]:
    """Replay all authored grammar and interaction evidence into the derived state table."""

    lessons = db.scalars(select(Lesson).order_by(Lesson.imported_at, Lesson.id)).all()
    documents = {lesson.id: lesson_document(lesson) for lesson in lessons}
    grammar = {lesson_id: _document_grammar(document) for lesson_id, document in documents.items()}

    catalog: dict[tuple[str, str, str], _Definition] = {}
    for lesson_id, document in documents.items():
        languages = (document.learning_language, document.translation_language)
        for construction_key, definition in grammar[lesson_id].definitions.items():
            key = (*languages, construction_key)
            previous = catalog.get(key)
            if previous is not None and previous != definition:
                raise ValueError(
                    "grammar construction has conflicting definitions across lessons: "
                    f"{construction_key}"
                )
            catalog[key] = definition

    interactions = db.scalars(
        select(Interaction).order_by(Interaction.occurred_at, Interaction.id)
    ).all()
    grouped: dict[tuple[int, str], list[Interaction]] = defaultdict(list)
    for interaction in interactions:
        grouped[(interaction.lesson_id, interaction.session_id)].append(interaction)

    evidence: list[_Evidence] = []
    for (lesson_id, _session_id), session_events in grouped.items():
        session_document = documents.get(lesson_id)
        lesson_grammar = grammar.get(lesson_id)
        if session_document is None or lesson_grammar is None:
            continue
        evidence.extend(
            _session_evidence(
                lesson_id,
                session_document,
                lesson_grammar,
                session_events,
                policy,
            )
        )
    evidence.sort(
        key=lambda item: (
            item.at,
            {"explicit_help": 0, "inferred_difficulty": 1, "success": 2}[item.kind],
            item.construction_key,
            item.lesson_id,
        )
    )

    states = {
        key: _State(
            definition=definition,
            learning_language=key[0],
            translation_language=key[1],
        )
        for key, definition in catalog.items()
    }
    for item in evidence:
        state = states.get(
            (item.learning_language, item.translation_language, item.construction_key)
        )
        if state is None:
            raise ValueError(
                f"grammar evidence references an unknown construction: {item.construction_key}"
            )
        if item.kind == "success":
            _apply_success(state, item, policy)
        else:
            _apply_failure(state, item, kind=item.kind, policy=policy)

    db.execute(delete(GrammarStateRow))
    rows = [_state_row(state) for _key, state in sorted(states.items())]
    db.add_all(rows)
    db.commit()
    return rows


def get_grammar_state(db: Session) -> GrammarState:
    """Return grammar from opened lessons, with recent contextual examples.

    Any interaction establishes that a lesson was opened, including data from older clients that
    did not emit ``lesson.started``. Merely importing or previewing prepared content is therefore
    insufficient to put a construction in this list.
    """

    profile = require_profile(db)
    opened_lessons, interactions = opened_lesson_interactions(db, profile_lessons(db, profile))
    if not opened_lessons:
        return _empty_grammar_state(profile)

    documents = {lesson.id: lesson_document(lesson) for lesson in opened_lessons}
    grammar = {lesson_id: _document_grammar(document) for lesson_id, document in documents.items()}
    exposure_times = lesson_exposure_times(interactions)

    aggregates: dict[str, _GrammarAggregate] = {}
    for lesson in opened_lessons:
        first_exposed, last_exposed = exposure_times[lesson.id]
        lesson_grammar = grammar[lesson.id]
        for sentence in lesson_grammar.sentences.values():
            for occurrence in sentence.occurrences:
                definition = lesson_grammar.definitions[occurrence.construction_key]
                aggregate = aggregates.setdefault(
                    occurrence.construction_key,
                    _GrammarAggregate(definition=definition),
                )
                aggregate.occurrence_count += 1
                aggregate.lesson_ids.add(lesson.id)
                aggregate.first_exposed_at = earliest_datetime(
                    aggregate.first_exposed_at,
                    first_exposed,
                )
                aggregate.last_exposed_at = latest_datetime(aggregate.last_exposed_at, last_exposed)
                aggregate.examples.append(
                    _ExampleCandidate(
                        exposed_at=last_exposed,
                        lesson_id=lesson.id,
                        title=documents[lesson.id].title,
                        sentence_key=sentence.key,
                        text=sentence.text,
                        translation=sentence.translation,
                        occurrence_key=occurrence.key,
                        note=occurrence.note,
                    )
                )

    if not aggregates:
        return _empty_grammar_state(profile)
    states = _grammar_states(db, profile, aggregates.keys())
    if aggregates.keys() - states.keys():
        # Recover once for databases whose derived cache predates grammar tracking; imports
        # and accepted evidence keep the cache current otherwise.
        rebuild_grammar_states(db)
        states = _grammar_states(db, profile, aggregates.keys())
    missing_states = aggregates.keys() - states.keys()
    if missing_states:
        raise RuntimeError(f"grammar state is missing constructions: {sorted(missing_states)}")

    raw_help_counts: dict[str, int] = defaultdict(int)
    help_sessions: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for interaction in interactions:
        if (
            interaction.event_type != "translation.revealed"
            or interaction.payload.get("scope") != "grammar"
        ):
            continue
        construction_key = interaction.payload.get("construction_key")
        if not isinstance(construction_key, str) or construction_key not in aggregates:
            continue
        raw_help_counts[construction_key] += 1
        help_sessions[construction_key].add((interaction.lesson_id, interaction.session_id))

    constructions = [
        _grammar_view(
            construction_key,
            aggregate,
            states[construction_key],
            raw_help_count=raw_help_counts[construction_key],
            counted_help_sessions=len(help_sessions[construction_key]),
        )
        for construction_key, aggregate in aggregates.items()
    ]
    constructions.sort(
        key=lambda construction: (
            construction.difficulty,
            construction.label.casefold(),
            construction.key,
        )
    )
    return GrammarState(
        learning_language=profile.learning_language,
        translation_language=profile.translation_language,
        constructions=constructions,
    )


def _empty_grammar_state(profile: Profile) -> GrammarState:
    return GrammarState(
        learning_language=profile.learning_language,
        translation_language=profile.translation_language,
        constructions=[],
    )


def _grammar_states(
    db: Session, profile: Profile, construction_keys: Collection[str]
) -> dict[str, GrammarStateRow]:
    return {
        state.construction_key: state
        for state in db.scalars(
            select(GrammarStateRow).where(
                GrammarStateRow.learning_language == profile.learning_language,
                GrammarStateRow.translation_language == profile.translation_language,
                GrammarStateRow.construction_key.in_(construction_keys),
            )
        )
    }


def _grammar_view(
    construction_key: str,
    aggregate: _GrammarAggregate,
    state: GrammarStateRow,
    *,
    raw_help_count: int,
    counted_help_sessions: int,
) -> GrammarView:
    first_exposed_at = aggregate.first_exposed_at
    last_exposed_at = aggregate.last_exposed_at
    if first_exposed_at is None or last_exposed_at is None:
        raise RuntimeError(f"grammar construction has no exposure time: {construction_key}")
    examples = sorted(
        aggregate.examples,
        key=lambda example: (
            example.exposed_at,
            example.lesson_id,
            example.sentence_key,
            example.occurrence_key,
        ),
        reverse=True,
    )[:3]
    return GrammarView(
        key=construction_key,
        label=aggregate.definition.label,
        description=aggregate.definition.description,
        category=aggregate.definition.category,
        difficulty=aggregate.definition.difficulty,
        occurrence_count=aggregate.occurrence_count,
        exposed_lesson_count=len(aggregate.lesson_ids),
        raw_help_count=raw_help_count,
        counted_help_sessions=counted_help_sessions,
        inferred_difficulty_signals=state.inferred_difficulty_signals,
        qualified_exposures=state.qualified_exposures,
        mastery=state.mastery,
        stability_days=state.stability_days,
        first_exposed_at=first_exposed_at,
        last_exposed_at=last_exposed_at,
        last_helped_at=optional_utc(state.last_helped_at),
        next_due_at=optional_utc(state.next_due_at),
        examples=[
            GrammarExampleView(
                lesson_id=example.lesson_id,
                title=example.title,
                sentence_key=example.sentence_key,
                text=example.text,
                translation=example.translation,
                note=example.note,
            )
            for example in examples
        ],
    )


def _session_evidence(
    lesson_id: int,
    document: LessonDocument,
    grammar: _DocumentGrammar,
    events: Sequence[Interaction],
    policy: GrammarLearningPolicy,
) -> list[_Evidence]:
    qualified_completion = next(
        (
            event
            for event in events
            if event.event_type == "lesson.completed"
            and payload_is_qualified_reading(event.payload, policy.qualification)
        ),
        None,
    )
    cutoff = (
        (as_utc(qualified_completion.occurred_at), qualified_completion.id)
        if qualified_completion is not None
        else None
    )
    considered = [
        event
        for event in events
        if cutoff is None or (as_utc(event.occurred_at), event.id) <= cutoff
    ]

    explicit: dict[str, datetime] = {}
    inferred: dict[str, datetime] = {}
    revealed_terms: set[str] = set()
    translated_constructions: set[str] = set()
    full_translation = False
    completion_time: datetime | None = None

    for event in considered:
        at = as_utc(event.occurred_at)
        if event.event_type == "term.revealed":
            term_key = event.payload.get("term_key")
            if isinstance(term_key, str):
                revealed_terms.add(term_key)
            continue
        if event.event_type == "lesson.completed" and payload_is_qualified_reading(
            event.payload, policy.qualification
        ):
            completion_time = at
            continue
        if event.event_type != "translation.revealed":
            continue

        scope = event.payload.get("scope")
        if scope == "lesson":
            full_translation = True
        elif scope == "grammar":
            construction_key = _validated_help_reference(event.payload, grammar)
            explicit.setdefault(construction_key, at)
            translated_constructions.add(construction_key)
        elif scope == "sentence":
            sentence_key = event.payload.get("sentence_key")
            if not isinstance(sentence_key, str) or sentence_key not in grammar.sentences:
                raise ValueError(f"grammar evidence references an unknown sentence: {sentence_key}")
            sentence = grammar.sentences[sentence_key]
            construction_keys = {occurrence.construction_key for occurrence in sentence.occurrences}
            translated_constructions.update(construction_keys)
            vocabulary_reveals = len(revealed_terms & sentence.term_keys)
            if vocabulary_reveals <= policy.max_inferred_vocabulary_reveals:
                for construction_key in construction_keys:
                    inferred.setdefault(construction_key, at)

    result = [
        _evidence(lesson_id, document, key, at, "explicit_help") for key, at in explicit.items()
    ]
    result.extend(
        _evidence(lesson_id, document, key, at, "inferred_difficulty")
        for key, at in inferred.items()
        if key not in explicit
    )

    if completion_time is None or full_translation:
        return result
    helped = translated_constructions | explicit.keys()
    result.extend(
        _evidence(lesson_id, document, construction_key, completion_time, "success")
        for construction_key in grammar.construction_keys - helped
    )
    return result


def _validated_help_reference(payload: Mapping[str, object], grammar: _DocumentGrammar) -> str:
    construction_key = payload.get("construction_key")
    sentence_key = payload.get("sentence_key")
    occurrence_key = payload.get("occurrence_key")
    identifiers = (construction_key, sentence_key, occurrence_key)
    if not all(isinstance(value, str) for value in identifiers):
        raise ValueError("grammar help must identify its construction, sentence, and occurrence")
    construction_key = cast(str, construction_key)
    sentence_key = cast(str, sentence_key)
    occurrence_key = cast(str, occurrence_key)
    sentence = grammar.sentences.get(sentence_key)
    if sentence is None:
        raise ValueError(f"grammar help references an unknown sentence: {sentence_key}")
    if not any(
        occurrence.key == occurrence_key and occurrence.construction_key == construction_key
        for occurrence in sentence.occurrences
    ):
        raise ValueError(f"grammar help references an unknown occurrence: {occurrence_key}")
    return construction_key


def _evidence(
    lesson_id: int,
    document: LessonDocument,
    construction_key: str,
    at: datetime,
    kind: _EvidenceKind,
) -> _Evidence:
    return _Evidence(
        at=at,
        kind=kind,
        learning_language=document.learning_language,
        translation_language=document.translation_language,
        construction_key=construction_key,
        lesson_id=lesson_id,
    )


def _apply_failure(
    state: _State,
    evidence: _Evidence,
    *,
    kind: _FailureKind,
    policy: GrammarLearningPolicy,
) -> None:
    weight = (
        policy.explicit_help_weight
        if kind == "explicit_help"
        else policy.inferred_difficulty_weight
    )
    recall = estimated_recall(
        mastery=state.mastery,
        stability_days=state.stability,
        last_seen_at=state.last_seen,
        at=evidence.at,
    )
    state.beta += weight
    if kind == "explicit_help":
        state.explicit_help_failures += 1
        state.last_helped = evidence.at
    else:
        state.inferred_difficulty_signals += 1
    state.stability = stability_after_failure(
        stability_days=state.stability,
        recall=recall,
        penalty=weight,
        policy=policy.stability,
    )
    state.lesson_ids.add(evidence.lesson_id)
    state.first_seen = state.first_seen or evidence.at
    state.last_seen = evidence.at
    due_days = (
        policy.explicit_help_due_days
        if kind == "explicit_help"
        else policy.inferred_difficulty_due_days
    )
    state.next_due = evidence.at + timedelta(days=due_days)


def _apply_success(
    state: _State,
    evidence: _Evidence,
    policy: GrammarLearningPolicy,
) -> None:
    weight = policy.clean_success_weight
    if evidence.lesson_id not in state.successful_lesson_ids:
        weight += policy.new_lesson_success_bonus
    recall = (
        1.0
        if state.last_seen is None
        else estimated_recall(
            mastery=state.mastery,
            stability_days=state.stability,
            last_seen_at=state.last_seen,
            at=evidence.at,
            policy=policy.stability,
        )
    )
    state.stability = stability_after_weighted_success(
        stability_days=state.stability,
        recall=recall,
        evidence_mass=weight,
        gain_days_per_mass=policy.stability_gain_days_per_success_mass,
        policy=policy.stability,
    )
    state.alpha += weight
    state.qualified_exposures += 1
    state.lesson_ids.add(evidence.lesson_id)
    state.successful_lesson_ids.add(evidence.lesson_id)
    state.first_seen = state.first_seen or evidence.at
    state.last_seen = evidence.at
    state.next_due = evidence.at + timedelta(days=state.stability)


def _state_row(state: _State) -> GrammarStateRow:
    return GrammarStateRow(
        learning_language=state.learning_language,
        translation_language=state.translation_language,
        construction_key=state.definition.key,
        label=state.definition.label,
        description=state.definition.description,
        category=state.definition.category,
        difficulty=state.definition.difficulty,
        alpha=state.alpha,
        beta=state.beta,
        stability_days=state.stability,
        qualified_exposures=state.qualified_exposures,
        explicit_help_failures=state.explicit_help_failures,
        inferred_difficulty_signals=state.inferred_difficulty_signals,
        distinct_lessons=len(state.lesson_ids),
        first_seen_at=state.first_seen,
        last_seen_at=state.last_seen,
        last_helped_at=state.last_helped,
        next_due_at=state.next_due,
        updated_at=utc_now(),
    )


def _document_grammar(document: LessonDocument) -> _DocumentGrammar:
    """Normalize the schema-owned authored grammar contract for replay/read aggregation."""

    definitions = {
        construction_key: _normalize_definition(grammar_construction(construction_key))
        for construction_key in document.grammar_construction_keys()
    }

    sentences: dict[str, _Sentence] = {}
    for sentence in all_sentences(document):
        occurrences = tuple(_normalize_occurrence(raw) for raw in sentence.grammar)
        for occurrence in occurrences:
            if occurrence.construction_key not in definitions:
                raise ValueError(
                    "grammar occurrence references an unknown construction: "
                    f"{occurrence.construction_key}"
                )
        sentences[sentence.key] = _Sentence(
            key=sentence.key,
            text=sentence.text,
            translation=sentence.translation,
            term_keys=frozenset(run.term.key for run in sentence.runs if run.term is not None),
            occurrences=occurrences,
        )
    return _DocumentGrammar(definitions=definitions, sentences=sentences)


def _normalize_definition(raw: GrammarConstruction) -> _Definition:
    return _Definition(
        key=raw.key,
        label=raw.label,
        description=raw.description,
        category=raw.category,
        difficulty=raw.difficulty,
    )


def _normalize_occurrence(raw: GrammarOccurrence) -> _Occurrence:
    return _Occurrence(
        key=raw.key,
        construction_key=raw.construction_key,
        run_start=raw.run_start,
        run_end=raw.run_end,
        note=raw.note,
    )
