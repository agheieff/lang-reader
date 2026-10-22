"""The generation brief handed to the lesson-writing agent, and its target policy."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.character_word_support import character_adjusted_offer_score
from server.clock import as_utc, utc_now
from server.derived_state import ensure_derived_states
from server.generation_tasks import (
    GenerationMode,
)
from server.grammar_catalog import (
    GrammarConstruction,
    find_grammar_catalog,
)
from server.han import character_tracking_available
from server.language_packs import language_pack
from server.learning_policy import TERM_BAND_POLICY
from server.learning_units import LearningUnitIndex, build_learning_unit_index
from server.lesson_content import body_sentences, lesson_document
from server.lesson_queue import (
    skipped_lesson_ids,
)
from server.lexeme_learning import (
    DEFAULT_LEXEME_LEARNING_POLICY,
    learner_frontier_rank,
    learner_level_frontier_rank,
)
from server.memory_model import recall_now
from server.models import (
    CharacterState,
    GrammarState,
    Interaction,
    Lesson,
    LexemeState,
    ProficiencyState,
)
from server.profile_activation import (
    questionnaire_seed,
)
from server.profile_state import build_profile_view, ensure_profile, generation_mode
from server.reading_preferences import content_history, current_preferences
from server.schemas import (
    AgentBrief,
    AgentGrammarBrief,
    AgentGrammarCatalogEntry,
    AgentLessonBrief,
    AgentTermBrief,
    CalibrationGenerationBrief,
    FeedbackTag,
    GenerationTargetPolicy,
    LessonDocument,
    LessonSentence,
    ProficiencyView,
    ProfileView,
)
from server.session_feedback import (
    feedback_from_payload,
    latest_completed_session_feedback,
    latest_profile_feedback,
)
from server.vocabulary_plan import (
    DEFAULT_TARGET_KNOWN_SHARE,
    KnownShareInputs,
    known_share,
    plan_vocabulary,
)
from server.word_lists import word_list

CALIBRATION_PROBE_COUNT = 12
CALIBRATION_WORDS_PER_PROBE = 16
CALIBRATION_MIN_WORDS_PER_PROBE = 14
DEFAULT_GENERATED_TEXT_LENGTH = 300
MIN_GENERATED_TEXT_LENGTH = 50
MAX_GENERATED_TEXT_LENGTH = 2_000
WORDS_PER_DELIBERATE_TARGET = 180
MAX_PROTOCOL_TARGETS = 16
MAX_GENERATION_CANDIDATES = 64
DEFAULT_TARGET_KNOWN_RATIO = 0.85


def calibration_generation_brief(db: Session) -> CalibrationGenerationBrief | None:
    profile = ensure_profile(db)
    if generation_mode(profile) != "calibration":
        return None
    state = db.get(ProficiencyState, 1)
    target = (
        state.estimate if state is not None and state.estimate is not None else profile.difficulty
    )
    target = max(0.02, min(0.98, target))
    sequence = (state.qualified_attempts if state is not None else 0) + 1
    if sequence == 1:
        seed = questionnaire_seed(profile)
        if seed is None:
            start, stop = 0.10, 0.80
        else:
            mean, variance = seed
            radius = max(0.11, 2 * math.sqrt(variance))
            start, stop = _bounded_interval(mean, radius * 2)
        difficulties = _linspace(start, stop, CALIBRATION_PROBE_COUNT)
    else:
        difficulties = _linspace(
            max(0.02, target - 0.15),
            min(0.98, target + 0.15),
            CALIBRATION_PROBE_COUNT,
        )
    excluded = sorted(
        {
            probe.term_key
            for lesson in db.scalars(select(Lesson).order_by(Lesson.id))
            for document in [lesson_document(lesson)]
            if document.calibration is not None
            for probe in document.calibration.probes
        }
    )
    return CalibrationGenerationBrief(
        sequence=sequence,
        target_difficulty=round(target, 6),
        probe_difficulties=difficulties,
        excluded_term_keys=excluded,
    )


def _linspace(start: float, stop: float, count: int) -> list[float]:
    step = (stop - start) / (count - 1)
    return [round(start + step * index, 6) for index in range(count)]


def _bounded_interval(center: float, width: float) -> tuple[float, float]:
    width = min(0.96, max(0.22, width))
    start = min(max(0.02, center - width / 2), 0.98 - width)
    return start, start + width


def build_agent_brief(
    db: Session,
    *,
    feedback: Sequence[FeedbackTag] | None = None,
) -> AgentBrief:
    profile = ensure_profile(db)
    profile_view = build_profile_view(db, profile)
    profile_view = profile_view.model_copy(
        update={
            "preferences": {
                key: value for key, value in profile_view.preferences.items() if key != "tts"
            }
        }
    )
    ensure_derived_states(db)
    catalog = find_grammar_catalog(profile.learning_language)
    now = utc_now()
    lessons = db.scalars(
        select(Lesson)
        .where(
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
        )
        .order_by(Lesson.imported_at.desc(), Lesson.id.desc())
    ).all()
    documents = {lesson.id: lesson_document(lesson) for lesson in lessons}
    units = build_learning_unit_index(documents[lesson.id] for lesson in reversed(lessons))
    lesson_ids = [lesson.id for lesson in lessons]
    skipped_ids = skipped_lesson_ids(db, lesson_ids)
    interactions = (
        db.scalars(
            select(Interaction)
            .where(Interaction.lesson_id.in_(lesson_ids))
            .order_by(Interaction.occurred_at, Interaction.id)
        ).all()
        if lesson_ids
        else []
    )
    completion_by_lesson: set[int] = set()
    rating_by_lesson: dict[int, Literal[-1, 1]] = {}
    feedback_by_lesson: dict[int, list[FeedbackTag]] = {}
    for interaction in interactions:
        if interaction.event_type == "lesson.completed":
            completion_by_lesson.add(interaction.lesson_id)
        elif interaction.event_type == "lesson.rated":
            rating = interaction.payload.get("rating")
            if rating in {-1, 1}:
                rating_by_lesson[interaction.lesson_id] = rating
            feedback_by_lesson[interaction.lesson_id] = feedback_from_payload(interaction.payload)
    latest_feedback = (
        latest_completed_session_feedback(interactions) if feedback is None else list(feedback)
    )

    target_appearances: Counter[str] = Counter()
    for lesson_id, document in documents.items():
        if lesson_id in skipped_ids:
            continue
        target_appearances.update(
            {
                canonical
                for term_key in document.target_term_keys
                if (canonical := units.resolve(term_key)) is not None
            }
        )
    terms = db.scalars(
        select(LexemeState).where(
            LexemeState.learning_language == profile.learning_language,
            LexemeState.translation_language == profile.translation_language,
        )
    ).all()
    terms = [term for term in terms if term.term_key in units.definitions]
    establishment = {
        term.term_key: term.qualified_exposures + term.reveal_failures for term in terms
    }
    failure_lessons: dict[str, set[int]] = defaultdict(set)
    for interaction in interactions:
        if interaction.event_type != "term.revealed":
            continue
        term_key = interaction.payload.get("term_key")
        if isinstance(term_key, str):
            canonical = units.resolve(term_key)
            if canonical is not None:
                failure_lessons[canonical].add(interaction.lesson_id)
                continue
            selected = units.select_evidence_targets(term_key, establishment)
            if len(set(selected)) >= 2:
                for selected_key in selected:
                    failure_lessons[selected_key].add(interaction.lesson_id)
    scored = [
        (
            _urgency(
                term,
                now,
                target_appearances=target_appearances[term.term_key],
                failure_lesson_count=len(failure_lessons[term.term_key]),
            ),
            term,
        )
        for term in terms
    ]
    if character_tracking_available(profile.learning_language):
        character_states = {
            state.character: state
            for state in db.scalars(
                select(CharacterState).where(
                    CharacterState.learning_language == profile.learning_language,
                    CharacterState.translation_language == profile.translation_language,
                )
            ).all()
        }
        scored.sort(
            key=lambda item: (
                -character_adjusted_offer_score(
                    item[0],
                    item[1],
                    character_states,
                    at=now,
                ),
                item[1].frequency_rank or 10**9,
                item[1].term_key,
            )
        )
    else:
        scored.sort(key=lambda item: (-item[0], item[1].frequency_rank or 10**9, item[1].term_key))
    scored = _rotate_recent_skipped_offers(scored, lessons, units)
    selected_terms = _select_generation_candidates(
        scored,
        profile_view,
        latest_feedback,
    )
    priority_terms = [_agent_term(term, urgency, now) for urgency, term in selected_terms]
    priority_grammar = _select_priority_grammar(
        db,
        catalog.constructions if catalog is not None else (),
        lessons,
        documents,
        completion_by_lesson,
        skipped_ids,
        profile_view,
        latest_feedback,
        now,
    )
    share_inputs = KnownShareInputs(
        units=units,
        states={term.term_key: term for term in terms},
        frontier_rank=learner_frontier_rank(db, DEFAULT_LEXEME_LEARNING_POLICY),
        memory=DEFAULT_LEXEME_LEARNING_POLICY.memory,
        at=now,
    )
    recent_import_shares = [
        lesson.known_share_at_import
        for lesson in lessons
        if documents[lesson.id].calibration is None and lesson.known_share_at_import is not None
    ][:3]
    vocabulary = plan_vocabulary(
        entries=word_list(profile.learning_language),
        known_lemmas=(term.lemma for term in terms),
        frontier_rank=learner_level_frontier_rank(db),
        memory=share_inputs.memory,
        text_length=_effective_text_length(profile_view, latest_feedback),
        target_known_share=_target_known_share(profile_view.preferences),
        recent_known_shares=recent_import_shares,
    )
    mastered = sorted(
        term.term_key
        for term in terms
        if term.mastery >= TERM_BAND_POLICY.mastered_min_mastery
        and term.qualified_exposures >= TERM_BAND_POLICY.mastered_min_exposures
    )
    recent: list[AgentLessonBrief] = []
    for lesson in lessons[:10]:
        document = documents[lesson.id]
        sentences = body_sentences(document)
        recent.append(
            AgentLessonBrief(
                id=lesson.id,
                key=lesson.key,
                title=document.title,
                topic=document.topic,
                level=document.level,
                difficulty=document.difficulty,
                imported_at=as_utc(lesson.imported_at),
                completed=lesson.id in completion_by_lesson,
                rating=rating_by_lesson.get(lesson.id),
                feedback=feedback_by_lesson.get(lesson.id, []),
                target_term_keys=sorted(
                    {
                        canonical
                        for term_key in document.target_term_keys
                        if (canonical := units.resolve(term_key)) is not None
                    }
                ),
                grammar_keys=sorted(document.grammar_construction_keys()),
                opening_excerpt=_lesson_excerpt(sentences, from_end=False),
                ending_excerpt=_lesson_excerpt(sentences, from_end=True),
                calibration_lesson=document.calibration is not None,
                metadata=_agent_lesson_metadata(document.metadata),
                known_share=known_share(document, share_inputs),
            )
        )
    return AgentBrief(
        generated_at=now,
        profile=profile_view,
        lesson_count=len(lessons),
        completed_lesson_count=len(completion_by_lesson),
        interaction_count=len(interactions),
        language_guidance=list(language_pack(profile.learning_language).generation_guidance),
        grammar_catalog=(
            [agent_grammar_catalog_entry(construction) for construction in catalog.constructions]
            if catalog is not None
            else []
        ),
        priority_grammar=priority_grammar,
        priority_terms=priority_terms,
        mastered_term_keys=mastered,
        recent_lessons=recent,
        vocabulary=vocabulary,
        reading_preferences=current_preferences(db),
        content_history=content_history(db),
    )


_AGENT_LESSON_METADATA_KEYS = {
    "content_angle",
    "content_domain",
    "content_hypothesis",
    "content_plan",
    "continuity",
    "episode",
    "episode_number",
    "series",
    "series_key",
}


def _agent_lesson_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Expose semantic continuity while hiding stale generation bookkeeping."""

    return {key: value for key, value in metadata.items() if key in _AGENT_LESSON_METADATA_KEYS}


def _lesson_excerpt(sentences: Sequence[LessonSentence], *, from_end: bool) -> str:
    selected = sentences[-2:] if from_end else sentences[:2]
    text = " ".join(sentence.text.strip() for sentence in selected if sentence.text.strip())
    if len(text) <= 180:
        return text
    return text[:179].rstrip() + "…"


def build_generation_target_policy(
    db: Session,
    mode: GenerationMode,
    *,
    brief: AgentBrief | None = None,
    feedback: Sequence[FeedbackTag] | None = None,
) -> GenerationTargetPolicy:
    brief = brief or build_agent_brief(db, feedback=feedback)
    latest_feedback = latest_profile_feedback(db) if feedback is None else list(feedback)
    if mode == "calibration":
        return GenerationTargetPolicy(
            preferred_count=0,
            max_count=0,
            target_text_length=_calibration_text_length(brief.profile, latest_feedback),
            candidate_term_keys=[],
        )

    text_length = _effective_text_length(brief.profile, latest_feedback)
    candidate_keys = [term.key for term in brief.priority_terms]
    if not candidate_keys:
        return GenerationTargetPolicy(
            preferred_count=0,
            max_count=0,
            target_text_length=text_length,
            candidate_term_keys=[],
        )

    confidence = _learner_confidence(brief.profile.proficiency)
    effective_learning_words = text_length * (0.5 + confidence / 2.0)
    target_capacity = (
        effective_learning_words
        / WORDS_PER_DELIBERATE_TARGET
        * _profile_target_capacity_factor(brief.profile)
    )
    max_count = min(
        len(candidate_keys),
        MAX_PROTOCOL_TARGETS,
        math.ceil(target_capacity),
    )
    preferred_count = min(max_count, math.floor(target_capacity))
    return GenerationTargetPolicy(
        preferred_count=preferred_count,
        max_count=max_count,
        target_text_length=text_length,
        candidate_term_keys=candidate_keys,
    )


def agent_grammar_catalog_entry(
    construction: GrammarConstruction,
) -> AgentGrammarCatalogEntry:
    return AgentGrammarCatalogEntry(
        key=construction.key,
        label=construction.label,
        description=construction.description,
        category=construction.category,
        difficulty=construction.difficulty,
        generation_hint=construction.generation_hint,
    )


def _select_priority_grammar(
    db: Session,
    constructions: Sequence[GrammarConstruction],
    lessons: Sequence[Lesson],
    documents: Mapping[int, LessonDocument],
    completed_lesson_ids: set[int],
    skipped_lesson_ids: set[int],
    profile: ProfileView,
    feedback: Sequence[FeedbackTag],
    now: datetime,
) -> list[AgentGrammarBrief]:
    """Offer a small choice pool while leaving realization to the writing agent."""

    unread_keys = {
        construction_key
        for lesson in lessons
        if lesson.id not in completed_lesson_ids and lesson.id not in skipped_lesson_ids
        for construction_key in documents[lesson.id].grammar_construction_keys()
    }
    states = {
        state.construction_key: state
        for state in db.scalars(
            select(GrammarState).where(
                GrammarState.learning_language == profile.learning_language,
                GrammarState.translation_language == profile.translation_language,
            )
        )
    }
    cooled = _recent_skipped_grammar_offers(lessons)
    scored: list[tuple[float, AgentGrammarBrief]] = []
    for construction in constructions:
        if construction.key in unread_keys:
            continue
        state = states.get(construction.key)
        score, reason = _grammar_urgency(construction, state, profile.difficulty, now)
        if construction.key in cooled:
            score -= 0.18
        scored.append(
            (
                score,
                AgentGrammarBrief(
                    key=construction.key,
                    label=construction.label,
                    description=construction.description,
                    category=construction.category,
                    difficulty=construction.difficulty,
                    mastery=round(state.mastery, 6) if state is not None else 0.5,
                    stability_days=round(state.stability_days, 6) if state is not None else 0.5,
                    qualified_exposures=state.qualified_exposures if state is not None else 0,
                    help_failures=state.explicit_help_failures if state is not None else 0,
                    inferred_difficulty_signals=(
                        state.inferred_difficulty_signals if state is not None else 0
                    ),
                    next_due_at=(
                        as_utc(state.next_due_at)
                        if state is not None and state.next_due_at is not None
                        else None
                    ),
                    urgency=round(max(0.0, score), 6),
                    reason=reason,
                ),
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1].difficulty, item[1].key))
    if not scored:
        return []
    confidence = _learner_confidence(profile.proficiency)
    pool_size = max(3, math.ceil(math.sqrt(len(scored)) * (1.5 + confidence / 2)))
    if "more_grammar" in feedback:
        pool_size += 2
    elif "less_grammar" in feedback:
        pool_size -= 1
    return [brief for _score, brief in scored[: min(len(scored), max(1, pool_size), 12)]]


def _grammar_urgency(
    construction: GrammarConstruction,
    state: GrammarState | None,
    learner_difficulty: float,
    now: datetime,
) -> tuple[float, Literal["unseen", "due", "fragile"]]:
    distance = abs(construction.difficulty - learner_difficulty)
    level_fit = math.exp(-5.0 * distance)
    above_frontier = max(0.0, construction.difficulty - learner_difficulty - 0.18)
    frontier_fit = max(0.0, 1.0 - above_frontier * 4.0)
    if state is None or state.first_seen_at is None:
        return 0.45 + 0.30 * level_fit + 0.20 * frontier_fit, "unseen"

    if state.next_due_at is None:
        due = 0.45
        reason: Literal["unseen", "due", "fragile"] = "fragile"
    else:
        days_overdue = (now - as_utc(state.next_due_at)).total_seconds() / 86_400
        due = max(0.0, min(1.0, 0.5 + days_overdue / 7.0))
        reason = "due" if days_overdue >= 0 else "fragile"
    uncertainty = min(1.0, 4.0 / max(1.0, state.alpha + state.beta))
    signals = min(
        1.0,
        (state.explicit_help_failures + 0.35 * state.inferred_difficulty_signals) / 3.0,
    )
    sparse_evidence = 1.0 - min(1.0, state.qualified_exposures / 4.0)
    score = (
        0.30 * (1.0 - state.mastery)
        + 0.22 * due
        + 0.14 * uncertainty
        + 0.16 * signals
        + 0.10 * level_fit
        + 0.08 * sparse_evidence
    )
    return score, reason


def _recent_skipped_grammar_offers(lessons: Sequence[Lesson]) -> set[str]:
    for lesson in lessons:
        metadata = lesson_document(lesson).metadata
        value = metadata.get("grammar")
        if not isinstance(value, Mapping):
            value = metadata.get("grammar_targets")
        if not isinstance(value, Mapping):
            continue
        offered = _metadata_term_keys(value.get("offered"))
        realized = _metadata_term_keys(value.get("realized"))
        return offered - realized
    return set()


def _rotate_recent_skipped_offers(
    scored: Sequence[tuple[float, LexemeState]],
    lessons: Sequence[Lesson],
    units: LearningUnitIndex,
) -> list[tuple[float, LexemeState]]:
    cooled: set[str] = set()
    for lesson in lessons:
        targets = lesson_document(lesson).metadata.get("targets")
        if not isinstance(targets, Mapping):
            continue
        offered = {
            canonical
            for term_key in _metadata_term_keys(targets.get("offered"))
            if (canonical := units.resolve(term_key)) is not None
        }
        realized = {
            canonical
            for term_key in _metadata_term_keys(targets.get("realized"))
            if (canonical := units.resolve(term_key)) is not None
        }
        cooled = offered - realized
        break
    if not cooled:
        return list(scored)
    return [item for item in scored if item[1].term_key not in cooled] + [
        item for item in scored if item[1].term_key in cooled
    ]


def _metadata_term_keys(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _select_generation_candidates(
    scored: Sequence[tuple[float, LexemeState]],
    profile: ProfileView,
    feedback: Sequence[FeedbackTag],
) -> list[tuple[float, LexemeState]]:
    available = len(scored)
    if available == 0:
        return []

    confidence = _learner_confidence(profile.proficiency)
    text_length = _effective_text_length(profile, feedback)
    lesson_units = max(1.0, text_length / WORDS_PER_DELIBERATE_TARGET)
    uncertainty_breadth = 2.0 - confidence
    pool_size = math.ceil(math.sqrt(available * lesson_units) * uncertainty_breadth)
    pool_size = min(available, MAX_GENERATION_CANDIDATES, max(1, pool_size))

    review = [item for item in scored if item[1].first_seen_at is not None]
    unseen = [item for item in scored if item[1].first_seen_at is None]
    known_ratio = _target_known_ratio(profile.preferences)
    exploration_share = (1.0 - known_ratio) * confidence
    unseen_slots = min(len(unseen), math.floor(pool_size * exploration_share))
    review_slots = min(len(review), pool_size - unseen_slots)

    selected = review[:review_slots] + unseen[:unseen_slots]
    selected_keys = {item[1].term_key for item in selected}
    for item in scored:
        if len(selected) >= pool_size:
            break
        if item[1].term_key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(item[1].term_key)
    return selected[:pool_size]


def _effective_text_length(
    profile: ProfileView,
    feedback: Sequence[FeedbackTag],
) -> int:
    configured = profile.preferences.get("text_length")
    if isinstance(configured, bool) or not isinstance(configured, int):
        requested = DEFAULT_GENERATED_TEXT_LENGTH
    else:
        requested = min(MAX_GENERATED_TEXT_LENGTH, max(MIN_GENERATED_TEXT_LENGTH, configured))
    if "longer" in feedback:
        requested = math.ceil(requested * 4 / 3)
    elif "shorter" in feedback:
        requested = math.floor(requested * 3 / 4)
    return min(MAX_GENERATED_TEXT_LENGTH, max(MIN_GENERATED_TEXT_LENGTH, requested))


def _calibration_text_length(
    profile: ProfileView,
    feedback: Sequence[FeedbackTag],
) -> int:
    target = _effective_text_length(profile, feedback)
    if "text_length" not in profile.preferences:
        target = CALIBRATION_PROBE_COUNT * CALIBRATION_WORDS_PER_PROBE
        if "longer" in feedback:
            target = math.ceil(target * 4 / 3)
        elif "shorter" in feedback:
            target = math.floor(target * 3 / 4)
    minimum = CALIBRATION_PROBE_COUNT * CALIBRATION_MIN_WORDS_PER_PROBE
    return max(minimum, target)


def _learner_confidence(proficiency: ProficiencyView) -> float:
    if proficiency.lower is not None and proficiency.upper is not None:
        return max(0.0, min(1.0, 1.0 - (proficiency.upper - proficiency.lower)))
    if proficiency.source == "self_reported":
        return 0.5
    return 0.0


def _target_known_ratio(preferences: Mapping[str, Any]) -> float:
    configured = preferences.get("target_known_ratio")
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        return DEFAULT_TARGET_KNOWN_RATIO
    return max(0.5, min(1.0, float(configured)))


def _target_known_share(preferences: Mapping[str, Any]) -> float:
    configured = preferences.get("target_known_share")
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        return DEFAULT_TARGET_KNOWN_SHARE
    return max(0.85, min(0.99, float(configured)))


def _profile_target_capacity_factor(profile: ProfileView) -> float:
    known_ratio = _target_known_ratio(profile.preferences)
    known_ratio_adjustment = (DEFAULT_TARGET_KNOWN_RATIO - known_ratio) / 2.0
    difficulty_adjustment = (profile.difficulty - 0.5) / 4.0
    return 1.0 + known_ratio_adjustment + difficulty_adjustment


DESIRED_RETENTION = 0.9


def _urgency(
    term: LexemeState,
    now: datetime,
    *,
    target_appearances: int = 0,
    failure_lesson_count: int = 0,
) -> float:
    """Rank a word for deliberate reuse by how likely it is forgotten right now.

    Reviewed words are scored by 1 - predicted recall, with a bonus once recall has fallen below
    the desired retention (due). Words never reviewed here are scored by how likely they are still
    unknown given frequency and level, so common words the learner surely knows are not drilled.
    """

    frequency = (
        1.0 / (1.0 + math.log10(max(1, term.frequency_rank)))
        if term.frequency_rank is not None
        else 0.25
    )
    memory = term.memory_state
    if memory is None:
        need = 1.0 - term.prior_known
        due = 0.5
    else:
        recall = recall_now(memory, prior=term.prior_known, at=now)
        need = 1.0 - recall
        due = 1.0 if recall < DESIRED_RETENTION else 0.0
    uncertainty = min(1.0, 4.0 / (term.alpha + term.beta))
    failure_spread = min(1.0, failure_lesson_count / 3.0)
    target_signal = min(1.0, float(target_appearances))
    rare_incidental_penalty = (
        0.12
        if term.reveal_failures == 1
        and target_appearances == 0
        and term.frequency_rank is not None
        and term.frequency_rank > 10_000
        else 0.0
    )
    score = (
        0.42 * need
        + 0.18 * due
        + 0.14 * frequency
        + 0.08 * uncertainty
        + 0.1 * failure_spread
        + 0.08 * target_signal
        - rare_incidental_penalty
    )
    return round(max(0.0, score), 6)


def _agent_term(term: LexemeState, urgency: float, now: datetime) -> AgentTermBrief:
    if term.first_seen_at is None:
        reason: Literal["unseen", "due", "fragile"] = "unseen"
    elif term.next_due_at is not None and as_utc(term.next_due_at) <= now:
        reason = "due"
    else:
        reason = "fragile"
    return AgentTermBrief(
        key=term.term_key,
        lemma=term.lemma,
        pos=term.pos,
        gloss=term.gloss,
        pronunciation=term.pronunciation,
        frequency_rank=term.frequency_rank,
        mastery=round(term.mastery, 6),
        stability_days=round(term.stability_days, 6),
        qualified_exposures=term.qualified_exposures,
        reveal_failures=term.reveal_failures,
        next_due_at=as_utc(term.next_due_at) if term.next_due_at else None,
        urgency=urgency,
        reason=reason,
    )
