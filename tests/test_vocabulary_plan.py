from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from server.agent_brief import build_agent_brief
from server.db import SCHEMA_VERSION, migrate_database
from server.learning import import_lesson
from server.lexeme_learning import LEXEME_MEMORY_POLICY
from server.models import Base
from server.vocabulary_plan import (
    MAX_LIST_CANDIDATES,
    plan_vocabulary,
    profile_known_share_inputs,
)
from server.word_lists import WordListEntry, word_list


def _entries(ranks: list[int]) -> list[WordListEntry]:
    return [WordListEntry(f"w{rank}", None, "gloss", rank, 1, "test") for rank in sorted(ranks)]


def _plan(**overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "entries": _entries([10, 400, 800, 1_000, 1_500, 2_000, 90_000]),
        "known_lemmas": ["w400"],
        "frontier_rank": 1_000.0,
        "memory": LEXEME_MEMORY_POLICY,
        "text_length": 300,
        "target_known_share": 0.95,
        "recent_known_shares": [],
    }
    arguments.update(overrides)
    return plan_vocabulary(**arguments)


def test_candidates_skip_known_obvious_and_far_too_rare_words() -> None:
    lemmas = [candidate.lemma for candidate in _plan().list_candidates]
    assert "w400" not in lemmas  # already encountered
    assert "w10" not in lemmas  # almost certainly known already
    assert "w90000" not in lemmas  # far beyond the frontier
    assert lemmas == sorted(lemmas, key=lambda lemma: int(lemma[1:]))  # most frequent first


def test_new_word_counts_follow_the_known_share_loop() -> None:
    neutral = _plan(recent_known_shares=[0.95])
    too_hard = _plan(recent_known_shares=[0.88])
    too_easy = _plan(recent_known_shares=[0.99])

    assert too_hard.list_new_words < neutral.list_new_words < too_easy.list_new_words
    assert too_hard.free_new_words == 1
    assert neutral.free_new_words == too_easy.free_new_words == 2
    assert neutral.recent_known_share == 0.95


def test_the_agent_keeps_adding_words_after_the_list_is_exhausted() -> None:
    exhausted = _plan(known_lemmas=[f"w{rank}" for rank in (400, 800, 1_000, 1_500, 2_000)])
    assert exhausted.list_exhausted and exhausted.list_candidates == []
    assert exhausted.list_new_words == 0 and exhausted.free_new_words >= 2

    no_list = _plan(entries=[])
    assert not no_list.list_exhausted and no_list.free_new_words >= 2


def test_chinese_list_is_frequency_ordered_and_bounded() -> None:
    entries = word_list("zh-Hans")
    assert len(entries) > 10_000
    assert entries[0].lemma == "的"
    assert word_list("de-DE") == ()
    plan = _plan(entries=entries, known_lemmas=[], frontier_rank=2_000.0)
    assert 0 < len(plan.list_candidates) <= MAX_LIST_CANDIDATES


def test_import_records_known_share_and_brief_carries_the_plan(
    db: Session, lesson_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())
    assert lesson.known_share_at_import is not None
    assert 0.0 <= lesson.known_share_at_import <= 1.0

    brief = build_agent_brief(db)
    assert brief.vocabulary is not None
    assert brief.vocabulary.list_candidates == []  # no list for Spanish yet
    assert brief.vocabulary.free_new_words >= 2
    assert brief.vocabulary.recent_known_share == lesson.known_share_at_import
    assert brief.recent_lessons[0].known_share is not None
    assert profile_known_share_inputs(db).states


def test_version_three_database_gains_the_known_share_column(tmp_path: Path) -> None:
    path = tmp_path / "v3.db"
    Base.metadata.create_all(create_engine(f"sqlite:///{path}"))
    with sqlite3.connect(path) as connection:  # the version 3 lessons table
        connection.execute("ALTER TABLE lessons DROP COLUMN known_share_at_import")
        connection.execute("PRAGMA user_version = 3")
    migrate_database(create_engine(f"sqlite:///{path}"))
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(lessons)")}
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert "known_share_at_import" in columns
