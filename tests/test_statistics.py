from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.learning import import_lesson, record_events
from server.models import Interaction, Lesson, LexemeState, ProficiencyState, Profile
from server.profile_state import update_profile
from server.statistics import get_statistics_summary


def _add_title(payload: dict[str, Any]) -> dict[str, Any]:
    payload["title"] = "Título"
    payload["title_sentence"] = {
        "key": f"{payload['key']}:title",
        "runs": [
            {
                "text": "Título",
                "term": {
                    "key": "es:titulo:NOUN",
                    "lemma": "título",
                    "pos": "NOUN",
                    "gloss": "title",
                    "frequency_rank": 1200,
                },
            }
        ],
        "translation": "Title",
    }
    return payload


def test_empty_statistics_expose_stable_thresholds_and_level_prior(db: Session) -> None:
    summary = get_statistics_summary(db)

    assert summary.learning_language == "es-ES"
    assert summary.translation_language == "en"
    assert summary.words.model_dump() == {
        "total": 0,
        "learning": 0,
        "expected": 0,
        "familiar": 0,
        "mastered": 0,
        "expected_min_mastery": 0.56,
        "familiar_min_mastery": 0.64,
        "mastered_min_mastery": 0.85,
    }
    assert summary.reading.model_dump() == {
        "texts_read": 0,
        "completed_sessions": 0,
        "total_active_seconds": 0.0,
        "recent_average_wpm": None,
        "recent_wpm_sessions": 0,
        "lifetime_average_wpm": None,
        "lifetime_wpm_sessions": 0,
        "recent_window_size": 10,
        "minimum_wpm_active_seconds": 30.0,
        "minimum_wpm_completion_ratio": 0.8,
    }
    assert summary.level.category == "B1"
    assert summary.level.value == pytest.approx(5 / 12)
    assert summary.level.source == "unknown"
    assert summary.level.status == "unstarted"


def test_word_mastery_buckets_use_tracked_words_and_exact_boundaries(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked_words = [
        SimpleNamespace(mastery=value)
        for value in (0.0, 0.559, 0.56, 0.639, 0.64, 0.849, 0.85, 1.0)
    ]
    monkeypatch.setattr(
        "server.statistics.get_words_state",
        lambda _db: SimpleNamespace(words=tracked_words),
    )

    words = get_statistics_summary(db).words

    assert words.total == 8
    assert (words.learning, words.expected, words.familiar, words.mastered) == (2, 2, 2, 2)


def test_reading_distinguishes_texts_and_rereads_and_deduplicates_latest_completion(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    first = import_lesson(db, _add_title(lesson_factory(key="first-statistics")))
    second = import_lesson(
        db,
        lesson_factory(
            key="second-statistics",
            terms=[
                ("es:uno:NOUN", "uno", "uno", "NOUN", "one", 10),
                ("es:dos:NOUN", "dos", "dos", "NOUN", "two", 20),
            ],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                first.id,
                "lesson.completed",
                event_id="first-session-early",
                session_id="first-session",
                payload={"active_seconds": 30, "completion_ratio": 1},
            ),
            event_factory(
                first.id,
                "lesson.completed",
                event_id="first-session-latest",
                session_id="first-session",
                payload={"active_seconds": 60, "completion_ratio": 1},
                seconds=1,
            ),
            event_factory(
                first.id,
                "lesson.completed",
                event_id="first-reread",
                session_id="first-reread",
                payload={"active_seconds": 40, "completion_ratio": 0.8},
                seconds=2,
            ),
            event_factory(
                second.id,
                "lesson.completed",
                event_id="second-short",
                session_id="second-session",
                payload={"active_seconds": 20, "completion_ratio": 1},
                seconds=3,
            ),
        ],
    )

    reading = get_statistics_summary(db).reading

    assert reading.texts_read == 2
    assert reading.completed_sessions == 3
    assert reading.total_active_seconds == 120
    assert reading.lifetime_wpm_sessions == 2
    assert reading.recent_wpm_sessions == 2
    assert reading.lifetime_average_wpm == pytest.approx(60 * (4 + 4 * 0.8) / (60 + 40))
    assert reading.recent_average_wpm == reading.lifetime_average_wpm


def test_wpm_is_qualified_weighted_and_uses_ten_latest_representative_sessions(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory(key="wpm-window"))
    events = [
        event_factory(
            lesson.id,
            "lesson.completed",
            event_id=f"qualified-{index}",
            session_id=f"qualified-{index}",
            payload={"active_seconds": 30 + index, "completion_ratio": 1},
            seconds=index,
        )
        for index in range(12)
    ]
    events.extend(
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="too-short",
                session_id="too-short",
                payload={"active_seconds": 29, "completion_ratio": 1},
                seconds=12,
            ),
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="too-incomplete",
                session_id="too-incomplete",
                payload={"active_seconds": 60, "completion_ratio": 0.79},
                seconds=13,
            ),
        ]
    )
    record_events(db, events)

    reading = get_statistics_summary(db).reading

    assert reading.completed_sessions == 14
    assert reading.total_active_seconds == sum(range(30, 42)) + 29 + 60
    assert reading.lifetime_wpm_sessions == 12
    assert reading.lifetime_average_wpm == pytest.approx(60 * 3 * 12 / sum(range(30, 42)))
    assert reading.recent_wpm_sessions == 10
    assert reading.recent_average_wpm == pytest.approx(60 * 3 * 10 / sum(range(32, 42)))


def test_missing_and_nonfinite_completion_values_contribute_no_time_or_wpm(
    db: Session, lesson_factory: Any, event_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory(key="invalid-statistics-time"))
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="missing-time",
                session_id="missing-time",
                payload={"completion_ratio": 1},
            )
        ],
    )
    db.add(
        Interaction(
            event_id="nonfinite-time",
            session_id="nonfinite-time",
            lesson_id=lesson.id,
            event_type="lesson.completed",
            occurred_at=datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
            payload={"active_seconds": float("nan"), "completion_ratio": 1},
        )
    )
    db.commit()

    reading = get_statistics_summary(db).reading

    assert reading.texts_read == 1
    assert reading.completed_sessions == 2
    assert reading.total_active_seconds == 0
    assert reading.recent_average_wpm is None
    assert reading.lifetime_average_wpm is None


def test_level_summary_exposes_one_current_snapshot(db: Session) -> None:
    update_profile(db, {"level": "B2", "difficulty": 0.6})
    state = db.get(ProficiencyState, 1)
    assert state is not None
    state.status = "rough"
    state.estimate = 0.58
    state.lower = 0.45
    state.upper = 0.71
    state.level = "B2"
    state.lower_level = "B1"
    state.upper_level = "C1"
    state.qualified_attempts = 2
    state.usable_probes = 18
    db.commit()

    level = get_statistics_summary(db).level

    assert level.model_dump() == {
        "value": 0.58,
        "category": "B2",
        "source": "self_reported",
        "status": "rough",
        "lower": 0.45,
        "upper": 0.71,
        "lower_category": "B1",
        "upper_category": "C1",
        "qualified_attempts": 2,
        "usable_probes": 18,
        "qualified_readings": 0,
    }


def test_statistics_endpoint_is_profile_scoped_and_creates_no_state(
    api_client: TestClient,
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    lesson = import_lesson(db, lesson_factory(key="statistics-api"))
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id="statistics-api-completion",
                payload={"active_seconds": 60, "completion_ratio": 1},
            )
        ],
    )
    profile = db.get(Profile, 1)
    assert profile is not None
    before = {
        "interactions": db.scalar(select(func.count()).select_from(Interaction)),
        "lessons": db.scalar(select(func.count()).select_from(Lesson)),
        "lexemes": db.scalar(select(func.count()).select_from(LexemeState)),
        "profile_updated_at": profile.updated_at,
    }

    response = api_client.get("/api/profiles/es-es/statistics")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "learning_language",
        "translation_language",
        "words",
        "reading",
        "level",
    }
    assert payload["learning_language"] == "es-ES"
    assert payload["reading"]["texts_read"] == 1
    assert payload["reading"]["lifetime_average_wpm"] == 3
    db.expire_all()
    profile = db.get(Profile, 1)
    assert profile is not None
    after = {
        "interactions": db.scalar(select(func.count()).select_from(Interaction)),
        "lessons": db.scalar(select(func.count()).select_from(Lesson)),
        "lexemes": db.scalar(select(func.count()).select_from(LexemeState)),
        "profile_updated_at": profile.updated_at,
    }
    assert after == before
