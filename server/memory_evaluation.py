"""Replay real reading history to score the vocabulary memory model against observed reveals.

At the first evidence of every qualified reading session, the model predicts P(recall) for each
learning unit observed in that session; the outcome is whether the learner revealed it before
finishing. Predictions precede the session's own evidence, so the score is one-step-ahead. The
replay drives the production reducer functions, so the score measures exactly what the app uses.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.learning_units import learning_unit_indexes
from server.lesson_content import lesson_document
from server.lexeme_learning import (
    DEFAULT_LEXEME_LEARNING_POLICY,
    LexemeLearningPolicy,
    _apply_failure,
    _apply_success,
    _LearningEvidence,
    _LexemeEstimate,
    _ordered_evidence,
    learner_frontier_rank,
)
from server.memory_model import prior_known, recall_now
from server.models import Interaction, Lesson
from server.schemas import LessonTerm


@dataclass(frozen=True)
class History:
    evidence: list[_LearningEvidence]
    rereads: set[int]  # ids of success evidence items that belong to a reread
    observed: dict[tuple[int, str], set[str]]
    terms: dict[str, LessonTerm]
    base_frontier_rank: float


@dataclass(frozen=True)
class Score:
    observations: int
    reveal_rate: float
    log_loss: float
    brier: float
    auc: float

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "observations": self.observations,
            "reveal_rate": round(self.reveal_rate, 4),
            "log_loss": round(self.log_loss, 4) if math.isfinite(self.log_loss) else None,
            "brier": round(self.brier, 4) if math.isfinite(self.brier) else None,
            "auc": round(self.auc, 4) if math.isfinite(self.auc) else None,
        }


def load_history(db: Session) -> History:
    lessons = db.scalars(select(Lesson).order_by(Lesson.imported_at, Lesson.id)).all()
    documents = {lesson.id: lesson_document(lesson) for lesson in lessons}
    interactions = db.scalars(
        select(Interaction).order_by(Interaction.occurred_at, Interaction.id)
    ).all()
    indexes = learning_unit_indexes(documents.values())
    policy = DEFAULT_LEXEME_LEARNING_POLICY
    ordered = [
        item
        for item in _ordered_evidence(documents, interactions, indexes, policy)
        if item.term_key is not None and item.kind != "exclude"
    ]
    rereads: set[int] = set()
    first_session: dict[int, str] = {}
    for position, item in enumerate(ordered):
        if (
            item.kind == "success"
            and first_session.setdefault(item.lesson_id, item.session_id) != item.session_id
        ):
            rereads.add(position)
    qualified = {(item.lesson_id, item.session_id) for item in ordered if item.kind == "success"}
    observed: dict[tuple[int, str], set[str]] = defaultdict(set)
    for item in ordered:
        session = (item.lesson_id, item.session_id)
        if session in qualified and item.term_key is not None:
            observed[session].add(item.term_key)
    terms = {
        key: term for index in indexes.values() for key, term in index.learning_catalog().items()
    }
    # Undo the configured scale so fitting can explore it independently.
    base = learner_frontier_rank(db, policy) / policy.frontier_scale
    return History(ordered, rereads, dict(observed), terms, base)


def evaluate(
    history: History,
    policy: LexemeLearningPolicy = DEFAULT_LEXEME_LEARNING_POLICY,
    *,
    sessions: set[tuple[int, str]] | None = None,
) -> Score:
    frontier = history.base_frontier_rank * policy.frontier_scale
    states: dict[str, _LexemeEstimate] = {}
    failures_seen: set[tuple[int, str, str]] = set()
    clicked = {
        (item.lesson_id, item.session_id, item.term_key)
        for item in history.evidence
        if item.kind == "failure"
    }
    predicted: set[tuple[int, str]] = set()
    rows: list[tuple[float, int]] = []

    def state(key: str) -> _LexemeEstimate:
        if key not in states:
            term = history.terms[key]
            states[key] = _LexemeEstimate(
                term=term,
                learning_language="",
                translation_language="",
                prior=prior_known(term.frequency_rank, frontier, policy.memory),
            )
        return states[key]

    for position, item in enumerate(history.evidence):
        assert item.term_key is not None
        session = (item.lesson_id, item.session_id)
        wanted = sessions is None or session in sessions
        if session in history.observed and session not in predicted:
            predicted.add(session)
            if wanted:
                for key in sorted(history.observed[session]):
                    estimate = state(key)
                    recall = recall_now(estimate.memory, prior=estimate.prior, at=item.at)
                    rows.append((recall, 0 if (*session, key) in clicked else 1))
        if item.kind == "failure":
            marker = (*session, item.term_key)
            if marker not in failures_seen:
                failures_seen.add(marker)
                _apply_failure(state(item.term_key), item, 1.0, policy)
        else:
            _apply_success(state(item.term_key), item, policy, reread=position in history.rereads)
    return _score(rows)


def fit(
    history: History,
    *,
    sessions: set[tuple[int, str]] | None = None,
) -> tuple[LexemeLearningPolicy, Score]:
    """Coarse grid over the reading-specific parameters; FSRS weights stay at their defaults."""

    base = DEFAULT_LEXEME_LEARNING_POLICY
    best: tuple[LexemeLearningPolicy, Score] | None = None
    for confidence, slope, scale in itertools.product(
        (0.15, 0.3, 0.45, 0.6, 0.8),
        (0.6, 0.8, 1.0, 1.3, 1.7),
        (0.125, 0.25, 0.5, 1.0, 2.0),
    ):
        policy = replace(
            base,
            memory=replace(base.memory, passive_confidence=confidence, prior_slope=slope),
            frontier_scale=scale,
        )
        score = evaluate(history, policy, sessions=sessions)
        if best is None or score.log_loss < best[1].log_loss:
            best = (policy, score)
    assert best is not None
    return best


def session_order(history: History) -> list[tuple[int, str]]:
    ordered: list[tuple[int, str]] = []
    for item in history.evidence:
        session = (item.lesson_id, item.session_id)
        if session in history.observed and session not in ordered:
            ordered.append(session)
    return ordered


def describe_policy(policy: LexemeLearningPolicy) -> dict[str, float]:
    return {
        "passive_confidence": policy.memory.passive_confidence,
        "prior_slope": policy.memory.prior_slope,
        "frontier_scale": policy.frontier_scale,
    }


def _score(predictions: Iterable[tuple[float, int]]) -> Score:
    rows = [(min(0.99, max(0.01, p)), outcome) for p, outcome in predictions]
    if not rows:
        return Score(0, 0.0, math.nan, math.nan, math.nan)
    log_loss = -sum(math.log(p) if outcome else math.log(1 - p) for p, outcome in rows) / len(rows)
    brier = sum((p - outcome) ** 2 for p, outcome in rows) / len(rows)
    reveal_rate = sum(1 - outcome for _p, outcome in rows) / len(rows)
    return Score(len(rows), reveal_rate, log_loss, brier, _auc(rows))


def _auc(rows: Sequence[tuple[float, int]]) -> float:
    positives = sum(outcome for _p, outcome in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        return math.nan
    ranked = sorted(rows, key=lambda row: row[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index
        while end + 1 < len(ranked) and ranked[end + 1][0] == ranked[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        rank_sum += average_rank * sum(outcome for _p, outcome in ranked[index : end + 1])
        index = end + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
