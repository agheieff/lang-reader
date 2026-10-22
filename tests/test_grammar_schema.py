from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from server.grammar_catalog import (
    find_grammar_catalog,
    grammar_catalog,
    grammar_catalog_digest,
    grammar_catalogs,
    grammar_construction,
    load_grammar_catalogs,
)
from server.schemas import GeneratedLessonDraft, InteractionIn, LessonDocument, ReaderProgress


def _grammar_occurrence(
    *,
    key: str = "grammar-1",
    construction_key: str = "es:present-indicative",
    run_start: int = 1,
    run_end: int = 2,
    note: str | None = "This verb presents the action as a current habit.",
) -> dict[str, Any]:
    occurrence: dict[str, Any] = {
        "key": key,
        "construction_key": construction_key,
        "run_start": run_start,
        "run_end": run_end,
    }
    if note is not None:
        occurrence["note"] = note
    return occurrence


def _term_run(text: str, key: str) -> dict[str, Any]:
    return {
        "text": text,
        "term": {
            "key": key,
            "lemma": text.casefold(),
            "pos": "word",
            "gloss": text,
            "frequency_rank": 100,
        },
    }


def _generated_draft() -> dict[str, Any]:
    return {
        "title": "Prueba",
        "title_sentence": {
            "key": "title-sentence",
            "runs": [_term_run("Prueba", "es:prueba:NOUN")],
            "translation": "Test",
            "grammar": [_grammar_occurrence(key="title-grammar", run_start=0, run_end=1)],
        },
        "topic": "testing",
        "level": "A1",
        "difficulty": 0.15,
        "blocks": [
            {
                "key": "block-1",
                "sentences": [
                    {
                        "key": "sentence-1",
                        "runs": [
                            _term_run("Leo", "es:leer:VERB"),
                            {"text": " "},
                            _term_run("mucho", "es:mucho:ADV"),
                            {"text": "."},
                        ],
                        "translation": "I read a lot.",
                    }
                ],
            }
        ],
        "target_term_keys": [],
    }


def test_tracked_grammar_catalogs_load_and_route_language_variants() -> None:
    catalogs = {catalog.id: catalog for catalog in grammar_catalogs()}

    assert set(catalogs) == {"de", "es", "zh"}
    assert grammar_catalog("es-419").id == "es"
    assert grammar_catalog("de-DE").id == "de"
    assert grammar_catalog("zh-Hant").id == "zh"
    assert grammar_construction("zh:le-perfective").category == "aspect"
    assert grammar_catalog_digest("es-ES") == grammar_catalog_digest("es-419")
    assert grammar_catalog_digest("es-ES") != grammar_catalog_digest("zh-Hans")
    assert find_grammar_catalog("fi-FI") is None
    assert all(len(catalog.constructions) <= 10 for catalog in catalogs.values())
    with pytest.raises(LookupError, match="no grammar catalog"):
        grammar_catalog("fi-FI")


def test_grammar_catalog_loader_rejects_unknown_fields_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    (tmp_path / "broken.toml").write_text(
        """
schema_version = 1
id = "es"
tags = ["es"]
description = "Broken catalog"
unknown = true
[[constructions]]
key = "es:demo"
label = "Demo"
description = "Demo construction"
category = "test"
difficulty = 0.2
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="unknown"):
        load_grammar_catalogs(sorted(tmp_path.glob("*.toml")))

    (tmp_path / "broken.toml").write_text(
        """
schema_version = 1
id = "es"
tags = ["es"]
description = "Broken catalog"
[[constructions]]
key = "es:demo"
label = "Demo"
description = "First definition"
category = "test"
difficulty = 0.2
[[constructions]]
key = "es:demo"
label = "Duplicate"
description = "Second definition"
category = "test"
difficulty = 0.3
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="must be unique"):
        load_grammar_catalogs(sorted(tmp_path.glob("*.toml")))


def test_manual_lessons_remain_compatible_and_occurrence_note_is_optional(
    lesson_factory: Any,
) -> None:
    legacy = LessonDocument.model_validate(lesson_factory())
    assert legacy.grammar_occurrences() == {}
    assert all(not sentence.grammar for block in legacy.blocks for sentence in block.sentences)

    payload = lesson_factory()
    payload["blocks"][0]["sentences"][0]["grammar"] = [_grammar_occurrence(note=None)]
    document = LessonDocument.model_validate(payload)

    occurrence = document.grammar_occurrences()["grammar-1"]
    assert occurrence.construction_key == "es:present-indicative"
    assert occurrence.note is None
    assert document.grammar_occurrence_sentences() == {"grammar-1": "sentence-1"}
    assert document.grammar_construction_keys() == {"es:present-indicative"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"run_start": 2, "run_end": 2}, "greater than run_start"),
        ({"run_start": 1, "run_end": 20}, "exceeds sentence runs"),
        ({"construction_key": "es:not-in-catalog"}, "unknown grammar construction"),
    ],
)
def test_grammar_occurrences_reject_invalid_ranges_and_unknown_constructions(
    lesson_factory: Any,
    change: dict[str, Any],
    message: str,
) -> None:
    payload = lesson_factory()
    occurrence = _grammar_occurrence()
    occurrence.update(change)
    payload["blocks"][0]["sentences"][0]["grammar"] = [occurrence]

    with pytest.raises(ValidationError, match=message):
        LessonDocument.model_validate(payload)


def test_lesson_rejects_duplicate_occurrence_keys_and_wrong_language_catalog(
    lesson_factory: Any,
) -> None:
    duplicate = lesson_factory()
    duplicate["blocks"][0]["sentences"][0]["grammar"] = [_grammar_occurrence()]
    duplicate["blocks"][0]["sentences"][1]["grammar"] = [
        _grammar_occurrence(run_start=0, run_end=1)
    ]
    with pytest.raises(ValidationError, match="unique across a lesson"):
        LessonDocument.model_validate(duplicate)

    wrong_language = lesson_factory(learning_language="de-DE")
    wrong_language["blocks"][0]["sentences"][0]["grammar"] = [_grammar_occurrence()]
    with pytest.raises(ValidationError, match="do not belong to learning language de-DE"):
        LessonDocument.model_validate(wrong_language)


def test_generated_drafts_support_grammar_and_enforce_lesson_unique_occurrence_keys() -> None:
    payload = _generated_draft()
    draft = GeneratedLessonDraft.model_validate(payload)
    assert draft.title_sentence.grammar[0].key == "title-grammar"

    duplicate = deepcopy(payload)
    duplicate["blocks"][0]["sentences"][0]["grammar"] = [
        _grammar_occurrence(key="title-grammar", run_start=0, run_end=1)
    ]
    with pytest.raises(ValidationError, match="unique across a lesson"):
        GeneratedLessonDraft.model_validate(duplicate)


def test_grammar_help_uses_translation_event_contract_and_reader_progress_default() -> None:
    event = InteractionIn.model_validate(
        {
            "event_id": "grammar-help-1",
            "session_id": "session-1",
            "lesson_id": 1,
            "type": "translation.revealed",
            "payload": {
                "scope": "grammar",
                "sentence_key": "sentence-1",
                "occurrence_key": "grammar-1",
                "construction_key": "es:present-indicative",
            },
        }
    )
    assert event.payload["scope"] == "grammar"
    assert ReaderProgress().revealed_grammar_occurrence_keys == []

    missing_reference = event.model_dump(mode="json")
    del missing_reference["payload"]["occurrence_key"]
    with pytest.raises(ValidationError, match="occurrence_key"):
        InteractionIn.model_validate(missing_reference)
