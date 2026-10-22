from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy.orm import Session

from server.agent_brief import build_agent_brief
from server.character_word_support import (
    character_adjusted_offer_score,
    character_offer_multiplier,
    character_retrievability,
    word_character_support,
)
from server.learning import import_lesson, record_events
from server.models import CharacterState, LexemeState
from server.profile_state import update_profile

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)


def _character(
    character: str,
    *,
    alpha: float = 6.0,
    beta: float = 2.0,
    stability_days: float = 1.0,
    seen: bool = True,
) -> CharacterState:
    evidence_at = NOW if seen else None
    return CharacterState(
        learning_language="zh-Hans",
        translation_language="en",
        character=character,
        alpha=alpha,
        beta=beta,
        stability_days=stability_days,
        memory_difficulty=5.0 if seen else None,
        first_evidence_at=evidence_at,
        last_evidence_at=evidence_at,
    )


def _term(
    lemma: str,
    *,
    qualified_exposures: int = 0,
    reveal_failures: float = 0.0,
) -> LexemeState:
    return LexemeState(
        learning_language="zh-Hans",
        translation_language="en",
        term_key=f"zh:{lemma}:noun",
        lemma=lemma,
        pos="noun",
        gloss="test",
        qualified_exposures=qualified_exposures,
        reveal_failures=reveal_failures,
    )


def test_character_retrievability_follows_the_memory_curve() -> None:
    state = _character("湖")  # stability 1 day: recall falls to 90% after one day

    assert character_retrievability(state, at=NOW) == pytest.approx(1.0)
    assert character_retrievability(state, at=NOW + timedelta(days=1)) == pytest.approx(0.9)
    assert character_retrievability(_character("新", seen=False), at=NOW) == 0.0


def test_word_support_uses_each_distinct_han_identity_and_weakest_character() -> None:
    term = _term("湖湖边")
    states = {"湖": _character("湖"), "边": _character("边", seen=False)}

    support = word_character_support(term, states, at=NOW)

    expected_geometric = math.sqrt(1.0 * 0.05)
    assert support == pytest.approx(0.7 * expected_geometric)


def test_unknown_characters_are_zero_and_non_han_lemmas_are_neutral() -> None:
    assert word_character_support(_term("湖边"), {}, at=NOW) == pytest.approx(0.035)
    assert word_character_support(_term("USB"), {}, at=NOW) is None
    assert character_offer_multiplier(_term("USB"), {}, at=NOW) == 1.0


def test_direct_word_evidence_fades_the_character_hint_to_zero() -> None:
    states = {"湖": _character("湖")}
    unseen_word = _term("湖")
    established_word = _term("湖", qualified_exposures=2, reveal_failures=1.0)

    assert character_offer_multiplier(unseen_word, states, at=NOW) > 1.0
    assert character_offer_multiplier(established_word, states, at=NOW) == 1.0
    assert character_adjusted_offer_score(0.5, established_word, states, at=NOW) == 0.5


def test_character_offer_adjustment_is_bounded() -> None:
    weak = character_offer_multiplier(_term("未知"), {}, at=NOW)
    strong_states = {
        "已": _character("已", alpha=1000.0, beta=1.0),
        "知": _character("知", alpha=1000.0, beta=1.0),
    }
    strong = character_offer_multiplier(_term("已知"), strong_states, at=NOW)

    assert 0.85 <= weak < 1.0
    assert 1.0 < strong <= 1.15


def test_chinese_brief_uses_support_for_order_but_preserves_urgency(
    db: Session,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    known = import_lesson(
        db,
        _lesson("known-character", [("known", "明", "known", "zh:known:noun")]),
    )
    import_lesson(
        db,
        _lesson(
            "candidate-words",
            [
                ("supported", "明天", "tomorrow", "zh:zz-supported:noun"),
                ("unknown", "海洋", "ocean", "zh:aa-unknown:noun"),
            ],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                known.id,
                "lesson.completed",
                event_id="known-complete",
                payload={"active_seconds": 30, "completion_ratio": 0.8},
            )
        ],
    )
    reading_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("server.agent_brief.utc_now", lambda: reading_time)

    priority = build_agent_brief(db).priority_terms
    candidates = [
        item for item in priority if item.key in {"zh:zz-supported:noun", "zh:aa-unknown:noun"}
    ]

    assert [item.key for item in candidates] == [
        "zh:zz-supported:noun",
        "zh:aa-unknown:noun",
    ]
    assert candidates[0].urgency == candidates[1].urgency


def _lesson(
    key: str,
    terms: list[tuple[str, str, str, str]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "key": key,
        "title": f"Lesson {key}",
        "learning_language": "zh-Hans",
        "translation_language": "en",
        "topic": "test",
        "level": "A1",
        "difficulty": 0.15,
        "blocks": [
            {
                "key": "body",
                "sentences": [
                    {
                        "key": sentence_key,
                        "runs": [
                            {
                                "text": lemma,
                                "term": {
                                    "key": term_key,
                                    "lemma": lemma,
                                    "pos": "noun",
                                    "gloss": gloss,
                                    "frequency_rank": 100,
                                },
                            },
                            {"text": "。"},
                        ],
                        "translation": gloss,
                    }
                    for sentence_key, lemma, gloss, term_key in terms
                ],
            }
        ],
        "target_term_keys": [],
        "metadata": {},
    }
