"""Read-only profile statistics derived from durable lessons and learning evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.learning_policy import TERM_BAND_POLICY, TermBandPolicy
from server.lesson_content import lesson_document, lexical_token_count
from server.models import Interaction, Lesson, Profile
from server.profile_state import build_profile_view
from server.reading_evidence import (
    DEFAULT_READING_EVIDENCE_POLICY,
    finite_number,
    is_qualified_reading,
)
from server.schemas import (
    StatisticsLevelSummary,
    StatisticsReadingSummary,
    StatisticsSummary,
    StatisticsWordsSummary,
    WordView,
)
from server.words import get_words_state

RECENT_WPM_WINDOW_SIZE = 10


@dataclass(frozen=True)
class _CompletedSession:
    lesson_id: int
    payload: Mapping[str, Any]


def get_statistics_summary(db: Session) -> StatisticsSummary:
    """Return a side-effect-free summary for the configured language profile."""

    profile = db.get(Profile, 1)
    if profile is None:
        raise LookupError("profile is not configured")
    profile_view = build_profile_view(db, profile)
    word_views = get_words_state(db).words
    sessions = _completed_sessions(db, profile.learning_language, profile.translation_language)
    token_counts = _lesson_token_counts(db, {session.lesson_id for session in sessions})

    total_active_seconds = sum(
        seconds for session in sessions if (seconds := _active_seconds(session.payload)) is not None
    )
    representative_sessions: list[tuple[float, float]] = []
    for session in sessions:
        seconds = _active_seconds(session.payload)
        ratio = _completion_ratio(session.payload)
        if (
            seconds is not None
            and ratio is not None
            and is_qualified_reading(
                active_seconds=seconds,
                completion_ratio=ratio,
            )
        ):
            representative_sessions.append((token_counts[session.lesson_id] * ratio, seconds))
    recent_sessions = representative_sessions[-RECENT_WPM_WINDOW_SIZE:]
    proficiency = profile_view.proficiency
    return StatisticsSummary(
        learning_language=profile.learning_language,
        translation_language=profile.translation_language,
        words=_word_summary(word_views, TERM_BAND_POLICY),
        reading=StatisticsReadingSummary(
            texts_read=len({session.lesson_id for session in sessions}),
            completed_sessions=len(sessions),
            total_active_seconds=total_active_seconds,
            recent_average_wpm=_weighted_wpm(recent_sessions),
            recent_wpm_sessions=len(recent_sessions),
            lifetime_average_wpm=_weighted_wpm(representative_sessions),
            lifetime_wpm_sessions=len(representative_sessions),
            recent_window_size=RECENT_WPM_WINDOW_SIZE,
            minimum_wpm_active_seconds=(DEFAULT_READING_EVIDENCE_POLICY.minimum_active_seconds),
            minimum_wpm_completion_ratio=(DEFAULT_READING_EVIDENCE_POLICY.minimum_completion_ratio),
        ),
        level=StatisticsLevelSummary(
            value=profile_view.difficulty,
            category=profile_view.level,
            source=proficiency.source,
            status=proficiency.status,
            lower=proficiency.lower,
            upper=proficiency.upper,
            lower_category=proficiency.lower_level,
            upper_category=proficiency.upper_level,
            qualified_attempts=proficiency.qualified_attempts,
            usable_probes=proficiency.usable_probes,
            qualified_readings=proficiency.qualified_readings,
        ),
    )


def _word_summary(
    word_views: list[WordView],
    policy: TermBandPolicy,
) -> StatisticsWordsSummary:
    learning = expected = familiar = mastered = 0
    for word in word_views:
        if word.mastery < policy.expected_min_mastery:
            learning += 1
        elif word.mastery < policy.familiar_min_mastery:
            expected += 1
        elif word.mastery < policy.mastered_min_mastery:
            familiar += 1
        else:
            mastered += 1
    return StatisticsWordsSummary(
        total=len(word_views),
        learning=learning,
        expected=expected,
        familiar=familiar,
        mastered=mastered,
        expected_min_mastery=policy.expected_min_mastery,
        familiar_min_mastery=policy.familiar_min_mastery,
        mastered_min_mastery=policy.mastered_min_mastery,
    )


def _completed_sessions(
    db: Session, learning_language: str, translation_language: str
) -> list[_CompletedSession]:
    session_rank = func.row_number().over(
        partition_by=(Interaction.lesson_id, Interaction.session_id),
        order_by=(Interaction.occurred_at.desc(), Interaction.id.desc()),
    )
    ranked = (
        select(
            Interaction.lesson_id.label("lesson_id"),
            Interaction.session_id.label("session_id"),
            Interaction.payload.label("payload"),
            Interaction.occurred_at.label("occurred_at"),
            Interaction.id.label("interaction_id"),
            session_rank.label("session_rank"),
        )
        .join(Lesson, Lesson.id == Interaction.lesson_id)
        .where(
            Lesson.learning_language == learning_language,
            Lesson.translation_language == translation_language,
            Interaction.event_type == "lesson.completed",
        )
        .subquery()
    )
    rows = db.execute(
        select(ranked.c.lesson_id, ranked.c.payload)
        .where(ranked.c.session_rank == 1)
        .order_by(ranked.c.occurred_at, ranked.c.interaction_id)
    ).all()
    return [
        _CompletedSession(
            lesson_id=lesson_id,
            payload=payload if isinstance(payload, Mapping) else {},
        )
        for lesson_id, payload in rows
    ]


def _lesson_token_counts(db: Session, lesson_ids: set[int]) -> dict[int, int]:
    if not lesson_ids:
        return {}
    return {
        lesson.id: lexical_token_count(lesson_document(lesson))
        for lesson in db.scalars(select(Lesson).where(Lesson.id.in_(lesson_ids))).all()
    }


def _active_seconds(payload: Mapping[str, Any]) -> float | None:
    value = finite_number(payload.get("active_seconds"))
    return value if value is not None and value >= 0 else None


def _completion_ratio(payload: Mapping[str, Any]) -> float | None:
    value = finite_number(payload.get("completion_ratio"))
    return value if value is not None and 0 <= value <= 1 else None


def _weighted_wpm(sessions: list[tuple[float, float]]) -> float | None:
    active_seconds = sum(seconds for _tokens, seconds in sessions)
    if active_seconds <= 0:
        return None
    return 60 * sum(tokens for tokens, _seconds in sessions) / active_seconds
