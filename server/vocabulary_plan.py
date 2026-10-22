"""Known-word share of texts and the new-vocabulary plan for the next generated text."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.clock import utc_now
from server.learning_units import LearningUnitIndex, build_learning_unit_index
from server.lesson_content import body_sentences, lesson_document
from server.lexeme_learning import DEFAULT_LEXEME_LEARNING_POLICY, learner_frontier_rank
from server.memory_model import MemoryPolicy, prior_known, recall_now
from server.models import Lesson, LexemeState, Profile
from server.schemas import LessonDocument, NewWordCandidate, VocabularyPlan, is_proper_noun_pos
from server.word_lists import WordListEntry, normalized_lemma

# Comfortable extensive reading needs roughly 95-98% known running words (Nation; Hu & Nation).
DEFAULT_TARGET_KNOWN_SHARE = 0.95
# Offer list words the learner probably does not know yet but that sit near their frontier. List
# ranks are true corpus ranks, so callers pass the level frontier without the LLM-rank correction.
CANDIDATE_PRIOR_RANGE = (0.15, 0.6)
MAX_LIST_CANDIDATES = 16
MAX_LIST_NEW_WORDS = 6
WORDS_PER_LIST_NEW_WORD = 100


@dataclass(frozen=True)
class KnownShareInputs:
    units: LearningUnitIndex
    states: Mapping[str, LexemeState]
    frontier_rank: float
    memory: MemoryPolicy
    at: datetime


def profile_known_share_inputs(db: Session) -> KnownShareInputs:
    """Current vocabulary knowledge of the profile; derived states must already be fresh."""

    profile = db.get(Profile, 1)
    query = select(Lesson).order_by(Lesson.imported_at, Lesson.id)
    if profile is not None:
        query = query.where(
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
        )
    units = build_learning_unit_index(lesson_document(lesson) for lesson in db.scalars(query))
    policy = DEFAULT_LEXEME_LEARNING_POLICY
    return KnownShareInputs(
        units=units,
        states={state.term_key: state for state in db.scalars(select(LexemeState))},
        frontier_rank=learner_frontier_rank(db, policy),
        memory=policy.memory,
        at=utc_now(),
    )


def known_share(document: LessonDocument, inputs: KnownShareInputs) -> float | None:
    """Predicted fraction of running lexical tokens the learner recalls right now."""

    total = 0.0
    count = 0
    for sentence in body_sentences(document):
        for run in sentence.runs:
            term = run.term
            if term is None or is_proper_noun_pos(term.pos):
                continue
            canonical = inputs.units.resolve(term.key)
            state = inputs.states.get(canonical) if canonical is not None else None
            if state is not None:
                total += recall_now(state.memory_state, prior=state.prior_known, at=inputs.at)
            else:
                total += prior_known(term.frequency_rank, inputs.frontier_rank, inputs.memory)
            count += 1
    return round(total / count, 4) if count else None


def plan_vocabulary(
    *,
    entries: Sequence[WordListEntry],
    known_lemmas: Iterable[str],
    frontier_rank: float,
    memory: MemoryPolicy,
    text_length: int,
    target_known_share: float,
    recent_known_shares: Sequence[float],
) -> VocabularyPlan:
    """Choose list words near the frontier and how many new words the next text should add.

    The agent also always adds a few new words of its own choosing: a deliberate random element,
    and the only source of new vocabulary once the list is exhausted or absent.
    """

    known = {normalized_lemma(lemma) for lemma in known_lemmas}
    low, high = CANDIDATE_PRIOR_RANGE
    candidates: list[NewWordCandidate] = []
    for entry in entries:
        if normalized_lemma(entry.lemma) in known:
            continue
        prior = prior_known(entry.frequency_rank, frontier_rank, memory)
        if not low <= prior <= high:
            continue
        candidates.append(
            NewWordCandidate(
                lemma=entry.lemma,
                pronunciation=entry.pronunciation,
                gloss=entry.gloss,
                frequency_rank=entry.frequency_rank,
                level=entry.level,
                source=entry.source,
            )
        )
        if len(candidates) >= MAX_LIST_CANDIDATES:
            break

    recent = (
        round(sum(recent_known_shares) / len(recent_known_shares), 4)
        if (recent_known_shares)
        else None
    )
    base = max(1, round(text_length / WORDS_PER_LIST_NEW_WORD))
    # Closed loop: texts that read easier than the target get more new words, harder ones fewer.
    gap = 0.0 if recent is None else recent - target_known_share
    scaled = round(base * (1.0 + 20.0 * gap))
    list_new_words = max(0, min(MAX_LIST_NEW_WORDS, scaled, len(candidates)))
    exhausted = bool(entries) and not candidates
    if not entries or exhausted:
        free_new_words = max(2, min(MAX_LIST_NEW_WORDS, scaled))
    else:
        free_new_words = 1 if gap < -0.03 else 2
    return VocabularyPlan(
        list_candidates=candidates,
        list_new_words=list_new_words,
        free_new_words=free_new_words,
        list_exhausted=exhausted,
        target_known_share=target_known_share,
        recent_known_share=recent,
    )
