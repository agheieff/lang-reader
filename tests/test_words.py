from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.learning import import_lesson, record_events
from server.models import Interaction
from server.profile_state import update_profile
from server.words import get_words_state


def test_words_omit_imported_lessons_until_an_interaction_is_recorded(
    api_client: TestClient,
    db: Session,
    lesson_factory: Any,
) -> None:
    import_lesson(db, lesson_factory())

    state = get_words_state(db)
    response = api_client.get("/api/profiles/es-es/words")

    assert state.learning_language == "es-ES"
    assert state.translation_language == "en"
    assert state.words == []
    assert response.status_code == 200
    assert response.json() == {
        "learning_language": "es-ES",
        "translation_language": "en",
        "words": [],
    }


def test_words_aggregate_title_surfaces_opened_lessons_and_reveal_sessions(
    api_client: TestClient,
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    first_payload = lesson_factory(
        terms=[
            ("es:ir:VERB", "voy", "ir", "VERB", "to go", 25),
            ("es:ir:VERB", "vamos", "ir", "VERB", "to go", 25),
            ("es:ir:VERB", "voy", "ir", "VERB", "to go", 25),
            ("es:casa:NOUN", "casa", "casa", "NOUN", "house", 100),
        ],
        targets=["es:ir:VERB"],
    )
    first_payload["title"] = "Mi viaje"
    first_payload["title_sentence"] = {
        "key": "title-one",
        "runs": [
            {
                "text": "Mi",
                "term": {
                    "key": "es:mi:DET",
                    "lemma": "mi",
                    "pos": "DET",
                    "gloss": "my",
                    "frequency_rank": 12,
                },
            },
            {"text": " "},
            {
                "text": "viaje",
                "term": {
                    "key": "es:viaje:NOUN",
                    "lemma": "viaje",
                    "pos": "NOUN",
                    "gloss": "trip",
                    "pronunciation": "BYA-heh",
                    "frequency_rank": 750,
                },
            },
        ],
        "translation": "My trip",
    }
    first = import_lesson(db, first_payload)
    second = import_lesson(
        db,
        lesson_factory(
            key="lesson-two",
            terms=[("es:ir:VERB", "fuimos", "ir", "VERB", "to go", 25)],
            targets=[],
        ),
    )

    events = [
        event_factory(first.id, "lesson.started", event_id="start-one"),
        event_factory(
            first.id,
            "term.revealed",
            event_id="reveal-one",
            payload={"term_key": "es:ir:VERB"},
            seconds=1,
        ),
        event_factory(
            first.id,
            "term.revealed",
            event_id="reveal-repeat",
            payload={"term_key": "es:ir:VERB"},
            seconds=2,
        ),
        event_factory(
            first.id,
            "lesson.completed",
            event_id="complete-one",
            payload={"active_seconds": 40, "completion_ratio": 1},
            seconds=40,
        ),
        event_factory(
            first.id,
            "lesson.started",
            event_id="start-two",
            session_id="session-2",
            seconds=60,
        ),
        event_factory(
            first.id,
            "term.revealed",
            event_id="reveal-new-session",
            session_id="session-2",
            payload={"term_key": "es:ir:VERB"},
            seconds=61,
        ),
    ]
    record_events(db, events)

    response = api_client.get("/api/profiles/es-es/words")
    assert response.status_code == 200
    words = {word["key"]: word for word in response.json()["words"]}

    assert set(words) == {"es:casa:NOUN", "es:ir:VERB", "es:mi:DET", "es:viaje:NOUN"}
    assert words["es:viaje:NOUN"]["pronunciation"] == "BYA-heh"
    assert words["es:viaje:NOUN"]["surface_forms"] == [{"text": "viaje", "occurrences": 1}]

    ir = words["es:ir:VERB"]
    assert ir["lemma"] == "ir"
    assert ir["pos"] == "VERB"
    assert ir["gloss"] == "to go"
    assert ir["surface_forms"] == [
        {"text": "voy", "occurrences": 2},
        {"text": "vamos", "occurrences": 1},
    ]
    assert ir["occurrence_count"] == 3
    assert ir["exposed_lesson_count"] == 1
    assert ir["raw_reveal_count"] == 3
    assert ir["counted_reveal_sessions"] == 2
    assert ir["reveal_failures"] == 2
    assert ir["qualified_exposures"] == 0

    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert datetime.fromisoformat(ir["first_exposed_at"]) == start
    assert datetime.fromisoformat(ir["last_exposed_at"]) == start + timedelta(seconds=60)
    assert datetime.fromisoformat(ir["last_revealed_at"]) == start + timedelta(seconds=61)

    assert words["es:mi:DET"]["qualified_exposures"] == 1
    assert words["es:casa:NOUN"]["qualified_exposures"] == 1

    record_events(
        db,
        [
            event_factory(
                second.id,
                "lesson.rated",
                event_id="later-interaction-without-start",
                session_id="session-3",
                payload={"rating": 1},
                seconds=90,
            )
        ],
    )
    updated = {word.key: word for word in get_words_state(db).words}["es:ir:VERB"]
    assert [form.model_dump() for form in updated.surface_forms] == [
        {"text": "voy", "occurrences": 2},
        {"text": "fuimos", "occurrences": 1},
        {"text": "vamos", "occurrences": 1},
    ]
    assert updated.occurrence_count == 4
    assert updated.exposed_lesson_count == 2
    assert updated.raw_reveal_count == 3


def test_words_resolve_exact_aliases_using_all_profile_lessons(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    canonical = import_lesson(
        db,
        lesson_factory(
            key="canonical-lesson",
            terms=[("es:ir:canonical", "voy", "ir", "verb", "to go", 25)],
            targets=[],
        ),
    )
    alias = import_lesson(
        db,
        lesson_factory(
            key="alias-lesson",
            terms=[("es:ir:alias", "fuimos", "ir", "VERB", "to go", 999)],
            targets=[],
        ),
    )

    record_events(
        db,
        [
            event_factory(alias.id, "lesson.started", event_id="start-alias"),
            event_factory(
                alias.id,
                "term.revealed",
                event_id="reveal-alias",
                payload={"term_key": "es:ir:alias"},
                seconds=1,
            ),
        ],
    )

    first = get_words_state(db).words
    assert len(first) == 1
    assert first[0].key == "es:ir:canonical"
    assert [form.model_dump() for form in first[0].surface_forms] == [
        {"text": "fuimos", "occurrences": 1}
    ]
    assert first[0].raw_reveal_count == 1
    assert first[0].counted_reveal_sessions == 1

    record_events(
        db,
        [
            event_factory(
                canonical.id,
                "lesson.started",
                event_id="start-canonical",
                session_id="canonical-session",
                seconds=60,
            ),
            event_factory(
                canonical.id,
                "term.revealed",
                event_id="reveal-canonical",
                session_id="canonical-session",
                payload={"term_key": "es:ir:canonical"},
                seconds=61,
            ),
        ],
    )

    combined = get_words_state(db).words
    assert len(combined) == 1
    word = combined[0]
    assert word.key == "es:ir:canonical"
    assert word.frequency_rank == 25
    assert [form.model_dump() for form in word.surface_forms] == [
        {"text": "fuimos", "occurrences": 1},
        {"text": "voy", "occurrences": 1},
    ]
    assert word.occurrence_count == 2
    assert word.exposed_lesson_count == 2
    assert word.raw_reveal_count == 2
    assert word.counted_reveal_sessions == 2

    reveals = db.scalars(
        select(Interaction)
        .where(Interaction.event_type == "term.revealed")
        .order_by(Interaction.occurred_at, Interaction.id)
    ).all()
    assert [event.payload["term_key"] for event in reveals] == [
        "es:ir:alias",
        "es:ir:canonical",
    ]


def test_words_omit_productive_chinese_combinations_without_changing_events(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    lesson = import_lesson(
        db,
        lesson_factory(
            learning_language="zh-Hans",
            terms=[
                ("zh:one-classifier", "一个", "一个", "quantifier", "one; a", 16),
                ("zh:one", "一", "一", "numeral", "one", 2),
                ("zh:classifier", "个", "个", "classifier", "general classifier", 12),
                ("zh:together", "一起", "一起", "adverb", "together", 120),
            ],
            targets=[],
        ),
    )
    record_events(
        db,
        [
            event_factory(lesson.id, "lesson.started", event_id="start-zh"),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="reveal-combination",
                payload={"term_key": "zh:one-classifier"},
                seconds=1,
            ),
        ],
    )

    words = {word.key: word for word in get_words_state(db).words}
    assert set(words) == {"zh:one", "zh:classifier", "zh:together"}
    assert all(word.raw_reveal_count == 0 for word in words.values())
    assert words["zh:one"].surface_forms[0].occurrences == 2
    assert words["zh:classifier"].surface_forms[0].occurrences == 2
    assert words["zh:one"].counted_reveal_sessions == 1
    assert words["zh:classifier"].counted_reveal_sessions == 1
    assert words["zh:one"].reveal_failures == 0.5
    assert words["zh:classifier"].reveal_failures == 0.5
    assert words["zh:together"].counted_reveal_sessions == 0

    stored = db.scalar(select(Interaction).where(Interaction.event_id == "reveal-combination"))
    assert stored is not None
    assert stored.payload == {"term_key": "zh:one-classifier"}


def test_words_attribute_composite_checks_to_the_current_established_sense(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    update_profile(db, {"learning_language": "zh-Hans"})
    laugh = ("zh:laugh", "笑", "笑", "verb", "laugh", 600)
    completed = ("zh:le:completed", "了", "了", "particle", "completed action", 5)
    changed = ("zh:le:changed", "了", "了", "particle", "change of state", 5)
    composite = ("zh:laughed", "笑了", "笑了", "verb", "laughed", 600)
    established = import_lesson(
        db,
        lesson_factory(
            key="established-components",
            learning_language="zh-Hans",
            terms=[laugh, completed],
            targets=[],
        ),
    )
    import_lesson(
        db,
        lesson_factory(
            key="unopened-homograph",
            learning_language="zh-Hans",
            terms=[changed],
            targets=[],
        ),
    )
    composite_lesson = import_lesson(
        db,
        lesson_factory(
            key="composite-check",
            learning_language="zh-Hans",
            terms=[composite],
            targets=[],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                established.id,
                "lesson.completed",
                event_id="establish-components",
                session_id="establish",
                payload={"active_seconds": 45, "completion_ratio": 1},
            ),
            event_factory(
                composite_lesson.id,
                "term.revealed",
                event_id="reveal-established-sense",
                session_id="composite",
                payload={"term_key": composite[0]},
                seconds=60,
            ),
        ],
    )

    words = {word.key: word for word in get_words_state(db).words}

    assert changed[0] not in words
    assert words[laugh[0]].counted_reveal_sessions == 1
    assert words[completed[0]].counted_reveal_sessions == 1
    assert words[laugh[0]].raw_reveal_count == 0
    assert words[completed[0]].raw_reveal_count == 0
    assert words[laugh[0]].occurrence_count == 2
    assert words[completed[0]].occurrence_count == 2


def test_words_link_same_lemma_senses_without_merging_learning_evidence(
    api_client: TestClient,
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    lesson = import_lesson(
        db,
        lesson_factory(
            terms=[
                ("es:bajo:adjective", "bajo", "bajo", "adjective", "low", 450),
                ("es:bajo:preposition", "bajo", "bajo", "preposition", "under", 450),
            ],
            targets=[],
        ),
    )
    import_lesson(
        db,
        lesson_factory(
            key="unopened-sense",
            terms=[("es:bajo:noun", "bajo", "bajo", "noun", "bass", 450)],
            targets=[],
        ),
    )
    record_events(
        db,
        [
            event_factory(lesson.id, "lesson.started", event_id="start"),
            event_factory(
                lesson.id,
                "term.revealed",
                event_id="reveal-adjective",
                payload={"term_key": "es:bajo:adjective"},
            ),
        ],
    )

    response = api_client.get("/api/profiles/es-es/words")
    assert response.status_code == 200
    words = {word["key"]: word for word in response.json()["words"]}

    adjective = words["es:bajo:adjective"]
    preposition = words["es:bajo:preposition"]
    assert adjective["related_senses"] == [
        {
            "key": "es:bajo:preposition",
            "pos": "preposition",
            "gloss": "under",
            "pronunciation": None,
        }
    ]
    assert preposition["related_senses"] == [
        {
            "key": "es:bajo:adjective",
            "pos": "adjective",
            "gloss": "low",
            "pronunciation": None,
        }
    ]
    assert "es:bajo:noun" not in words
    assert adjective["raw_reveal_count"] == 1
    assert preposition["raw_reveal_count"] == 0
