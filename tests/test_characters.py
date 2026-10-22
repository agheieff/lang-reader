from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session
from starlette.requests import Request

from server.characters import get_characters_state
from server.han import character_tracking_available, is_han_character
from server.learning import import_lesson, record_events
from server.profile_state import update_profile
from server.workspaces import WorkspaceRegistry


def _term(
    key: str,
    lemma: str,
    pos: str,
    gloss: str,
    *,
    pronunciation: str,
) -> dict[str, object]:
    return {
        "key": key,
        "lemma": lemma,
        "pos": pos,
        "gloss": gloss,
        "pronunciation": pronunciation,
        "frequency_rank": 100,
    }


def _chinese_lesson() -> dict[str, object]:
    lake = _term("zh:湖边:noun", "湖边", "noun", "lakeside", pronunciation="hú biān")
    temperature = _term(
        "zh:温度:noun",
        "温度",
        "noun",
        "temperature",
        pronunciation="wēn dù",
    )
    short_while = _term(
        "zh:一会儿:noun",
        "一会儿",
        "noun",
        "a short while",
        pronunciation="yí huìr",
    )
    laugh = _term("zh:哈哈:verb", "哈哈", "verb", "to laugh", pronunciation="hā hā")
    return {
        "schema_version": 1,
        "key": "characters-one",
        "title": "湖边的温度",
        "title_sentence": {
            "key": "title",
            "runs": [
                {"text": "湖边", "term": lake},
                {"text": "的"},
                {"text": "温度", "term": temperature},
            ],
            "translation": "Temperature by the lake",
        },
        "learning_language": "zh-Hans",
        "translation_language": "en",
        "topic": "weather",
        "level": "A1",
        "difficulty": 0.15,
        "blocks": [
            {
                "key": "body",
                "sentences": [
                    {
                        "key": "wait",
                        "runs": [
                            {"text": "一会儿", "term": short_while},
                            {"text": "，"},
                            {"text": "哈哈", "term": laugh},
                            {"text": "。〇后後"},
                        ],
                        "translation": "A little while; haha.",
                    },
                    {
                        "key": "shore",
                        "runs": [
                            {"text": "湖畔", "term": lake},
                            {"text": "。"},
                        ],
                        "translation": "By the lake.",
                    },
                ],
            }
        ],
        "target_term_keys": [],
        "metadata": {},
    }


def test_character_tracking_is_routed_by_the_chinese_language_pack() -> None:
    assert character_tracking_available("zh-Hans")
    assert character_tracking_available("zh-Hant")
    assert not character_tracking_available("es-ES")

    assert is_han_character("湖")
    assert is_han_character("〇")
    assert is_han_character("\U00020000")
    assert not is_han_character("こ")
    assert not is_han_character("A")
    assert not is_han_character("湖边")


def test_characters_page_is_a_chinese_only_left_rail_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server import main

    test_registry = WorkspaceRegistry(tmp_path)
    test_registry.create(
        "zh-hans",
        label="Chinese (Simplified)",
        learning_language="zh-Hans",
        translation_language="en",
    )
    monkeypatch.setattr(main, "registry", test_registry)
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})

    spanish = main.profile_page(request, "es-es", "characters").body.decode()
    chinese = main.profile_page(request, "zh-hans", "characters").body.decode()

    assert 'data-initial-view="reading"' in spanish
    assert "?view=characters" not in spanish
    assert 'data-initial-view="characters"' in chinese
    assert 'href="/p/zh-hans?view=characters"' in chinese
    assert 'aria-label="Characters"' in chinese


def test_non_chinese_profiles_return_an_empty_observational_state(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    lesson = import_lesson(db, lesson_factory())
    record_events(db, [event_factory(lesson.id, "lesson.started", event_id="opened")])

    state = get_characters_state(db)

    assert state.learning_language == "es-ES"
    assert state.translation_language == "en"
    assert state.characters == []


def test_characters_use_opened_title_and_body_text_and_conservative_reveals(
    db: Session,
    event_factory: Any,
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    lesson = import_lesson(db, _chinese_lesson())

    assert get_characters_state(db).characters == []

    record_events(
        db,
        [
            event_factory(lesson.id, "lesson.started", event_id="start"),
            # This legacy reveal is ambiguous because the key renders as both 湖边 and 湖畔.
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="ambiguous-lake",
                payload={"term_key": "zh:湖边:noun"},
                seconds=1,
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="title-lake",
                payload={"term_key": "zh:湖边:noun", "sentence_key": "title"},
                seconds=2,
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="laugh",
                payload={"term_key": "zh:哈哈:verb", "sentence_key": "wait"},
                seconds=3,
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="wait-one",
                payload={"term_key": "zh:一会儿:noun", "sentence_key": "wait"},
                seconds=4,
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="wait-repeat",
                payload={"term_key": "zh:一会儿:noun", "sentence_key": "wait"},
                seconds=5,
            ),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="wait-new-session",
                session_id="session-2",
                payload={"term_key": "zh:一会儿:noun", "sentence_key": "wait"},
                seconds=60,
            ),
        ],
    )

    characters = {item.character: item for item in get_characters_state(db).characters}

    assert set(characters) == {
        "一",
        "会",
        "儿",
        "哈",
        "后",
        "度",
        "温",
        "湖",
        "的",
        "畔",
        "边",
        "〇",
        "後",
    }
    assert characters["湖"].occurrence_count == 2
    assert characters["湖"].raw_reveal_count == 1
    assert characters["边"].raw_reveal_count == 1
    assert characters["畔"].raw_reveal_count == 0
    assert characters["湖"].distinct_word_contexts == 1
    assert characters["湖"].qualified_exposures == 0
    assert characters["湖"].inferred_failure_sessions == 1
    assert characters["湖"].inferred_failure_mass == pytest.approx(0.1)
    assert characters["湖"].mastery < 0.5
    # mastery projects recall a month ahead, so it never exceeds current recall
    assert characters["湖"].mastery <= characters["湖"].retrievability
    assert characters["湖"].mastery_uncertainty > 0
    assert characters["湖"].last_evidence_at is not None
    assert characters["湖"].next_due_at is not None
    assert characters["哈"].occurrence_count == 2
    assert characters["哈"].raw_reveal_count == 1
    assert characters["哈"].inferred_failure_mass == pytest.approx(0.2)
    assert characters["一"].raw_reveal_count == 3
    assert characters["一"].counted_reveal_sessions == 2
    assert characters["一"].inferred_failure_sessions == 2
    assert characters["会"].raw_reveal_count == 3
    assert characters["儿"].raw_reveal_count == 3
    assert characters["后"].occurrence_count == 1
    assert characters["後"].occurrence_count == 1
    assert all(item.exposed_lesson_count == 1 for item in characters.values())
    assert all(item.direct_successes == 0 for item in characters.values())
    assert all(item.direct_failures == 0 for item in characters.values())

    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert characters["湖"].first_exposed_at == start
    assert characters["湖"].last_exposed_at == start + timedelta(seconds=60)
