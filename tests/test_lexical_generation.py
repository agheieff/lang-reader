from __future__ import annotations

from typing import Any

import pytest

from server.lexical_generation import (
    LexicalUnit,
    discover_lexical_conflicts,
    enumerate_lexical_units,
    merge_lexical_unit_results,
    validate_lexical_unit_result,
)
from server.schemas import (
    GenerationLexicalConflictResult,
    GenerationLexicalUnitResult,
    GenerationProseResult,
    LessonTerm,
)


def _prose_result(
    *,
    title: str = "顺序",
    blocks: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
        ("block-a", (("sentence-a", "甲。"), ("sentence-b", "乙。"))),
        ("block-b", (("sentence-c", "丙。"),)),
    ),
) -> GenerationProseResult:
    return GenerationProseResult.model_validate(
        {
            "schema_version": 1,
            "lessons": [
                {
                    "title": title,
                    "title_sentence": {"key": "title", "text": title},
                    "topic": "ordering",
                    "level": "A1",
                    "difficulty": 0.1,
                    "blocks": [
                        {
                            "key": block_key,
                            "sentences": [
                                {"key": sentence_key, "text": text}
                                for sentence_key, text in sentences
                            ],
                        }
                        for block_key, sentences in blocks
                    ],
                }
            ],
        }
    )


def _term(
    key: str,
    *,
    lemma: str,
    gloss: str,
    pronunciation: str,
    pos: str = "noun",
    frequency_rank: int = 100,
) -> LessonTerm:
    return LessonTerm(
        key=key,
        lemma=lemma,
        pos=pos,
        gloss=gloss,
        pronunciation=pronunciation,
        frequency_rank=frequency_rank,
    )


def _unit_result(
    unit: LexicalUnit,
    term: LessonTerm,
) -> GenerationLexicalUnitResult:
    text = unit.sentence.text
    runs: list[dict[str, str | None]]
    if text.endswith("。"):
        runs = [
            {"text": text[:-1], "term_key": term.key, "pronunciation": None},
            {"text": "。", "term_key": None, "pronunciation": None},
        ]
    else:
        runs = [{"text": text, "term_key": term.key, "pronunciation": None}]
    return GenerationLexicalUnitResult.model_validate(
        {
            "schema_version": 1,
            "unit_id": unit.unit_id,
            "key": unit.sentence.key,
            "terms": [term],
            "runs": runs,
        }
    )


def _two_unit_homograph_fixture() -> tuple[
    GenerationProseResult,
    tuple[LexicalUnit, ...],
]:
    prose = _prose_result(
        title="行",
        blocks=(("body", (("body-sentence", "行。"),)),),
    )
    return prose, enumerate_lexical_units(prose, "zh-Hans")


def test_enumerate_lexical_units_uses_title_then_canonical_block_order() -> None:
    prose = _prose_result()

    units = enumerate_lexical_units(prose, "zh-Hans")

    assert [
        (
            unit.lesson_index,
            unit.is_title,
            unit.block_index,
            unit.sentence.key,
            unit.sentence.text,
        )
        for unit in units
    ] == [
        (0, True, None, "title", "顺序"),
        (0, False, 0, "sentence-a", "甲。"),
        (0, False, 0, "sentence-b", "乙。"),
        (0, False, 1, "sentence-c", "丙。"),
    ]
    assert len({unit.unit_id for unit in units}) == len(units)
    assert all(unit.learning_language == "zh-Hans" for unit in units)


def test_validate_lexical_unit_result_rejects_exact_source_drift() -> None:
    unit = enumerate_lexical_units(_prose_result(), "zh-Hans")[0]
    term = _term(
        "local-1",
        lemma="别序",
        gloss="different order",
        pronunciation="bié xù",
    )
    drifted = GenerationLexicalUnitResult.model_validate(
        {
            "schema_version": 1,
            "unit_id": unit.unit_id,
            "key": unit.sentence.key,
            "terms": [term],
            "runs": [
                {"text": "别序", "term_key": term.key, "pronunciation": None},
            ],
        }
    )

    with pytest.raises(ValueError, match="source|frozen|reconstruct"):
        validate_lexical_unit_result(unit, drifted, "zh-Hans")


def test_same_local_key_with_distinct_unit_identities_gets_distinct_host_keys() -> None:
    prose, units = _two_unit_homograph_fixture()
    results = (
        _unit_result(
            units[0],
            _term("local-1", lemma="行", gloss="row", pronunciation="háng"),
        ),
        _unit_result(
            units[1],
            _term(
                "local-1",
                lemma="行",
                gloss="to walk",
                pronunciation="xíng",
                pos="verb",
            ),
        ),
    )
    groups = discover_lexical_conflicts(
        units,
        results,
        learning_language="zh-Hans",
        stored_terms=(),
    )
    conservative = GenerationLexicalConflictResult(
        schema_version=1,
        resolutions=[
            {
                "candidate_id": candidate.candidate_id,
                "canonical_candidate_id": candidate.candidate_id,
            }
            for group in groups
            for candidate in group.candidates
            if not candidate.stored
        ],
    )

    merged = merge_lexical_unit_results(
        prose,
        results,
        task_id=7,
        learning_language="zh-Hans",
        stored_terms=(),
        conflict_result=conservative,
    )

    lesson = merged.lessons[0]
    title_key = lesson.title_sentence.runs[0].term_key
    body_key = lesson.blocks[0].sentences[0].runs[0].term_key
    assert title_key is not None
    assert body_key is not None
    assert title_key != body_key
    assert {term.key for term in lesson.terms} == {title_key, body_key}


def test_exact_generated_identities_merge_across_unit_local_keys() -> None:
    prose, units = _two_unit_homograph_fixture()
    results = (
        _unit_result(
            units[0],
            _term("first-local", lemma="行", gloss="row", pronunciation="háng"),
        ),
        _unit_result(
            units[1],
            _term("second-local", lemma="行", gloss="row", pronunciation="háng"),
        ),
    )

    merged = merge_lexical_unit_results(
        prose,
        results,
        task_id=8,
        learning_language="zh-Hans",
        stored_terms=(),
    )

    lesson = merged.lessons[0]
    title_key = lesson.title_sentence.runs[0].term_key
    body_key = lesson.blocks[0].sentences[0].runs[0].term_key
    assert title_key == body_key
    assert len(lesson.terms) == 1


def test_conflict_resolution_can_merge_harmless_pos_label_drift() -> None:
    prose = _prose_result(
        title="能",
        blocks=(("body", (("body-sentence", "能。"),)),),
    )
    units = enumerate_lexical_units(prose, "zh-Hans")
    results = (
        _unit_result(
            units[0],
            _term(
                "verb-local",
                lemma="能",
                gloss="can; be able to",
                pronunciation="néng",
                pos="verb",
            ),
        ),
        _unit_result(
            units[1],
            _term(
                "modal-local",
                lemma="能",
                gloss="can; be able to",
                pronunciation="néng",
                pos="modal verb",
            ),
        ),
    )
    groups = discover_lexical_conflicts(
        units,
        results,
        learning_language="zh-Hans",
        stored_terms=(),
    )
    generated = [candidate for candidate in groups[0].candidates if not candidate.stored]
    canonical = generated[0].candidate_id
    resolution = GenerationLexicalConflictResult(
        schema_version=1,
        resolutions=[
            {
                "candidate_id": candidate.candidate_id,
                "canonical_candidate_id": canonical,
            }
            for candidate in generated
        ],
    )

    merged = merge_lexical_unit_results(
        prose,
        results,
        task_id=81,
        learning_language="zh-Hans",
        stored_terms=(),
        conflict_result=resolution,
    )

    assert len(merged.lessons[0].terms) == 1
    assert (
        merged.lessons[0].title_sentence.runs[0].term_key
        == merged.lessons[0].blocks[0].sentences[0].runs[0].term_key
    )


def test_exact_stored_identity_restores_the_complete_stored_definition_and_key() -> None:
    prose, units = _two_unit_homograph_fixture()
    stored = _term(
        "stored:hang",
        lemma="行",
        gloss="row",
        pronunciation="háng",
        frequency_rank=321,
    )
    generated = _term(
        "local-1",
        lemma="行",
        gloss="row",
        pronunciation="háng",
        frequency_rank=999,
    )
    results = tuple(_unit_result(unit, generated) for unit in units)

    merged = merge_lexical_unit_results(
        prose,
        results,
        task_id=9,
        learning_language="zh-Hans",
        stored_terms=(stored,),
    )

    lesson = merged.lessons[0]
    assert lesson.terms == [stored]
    assert lesson.title_sentence.runs[0].term_key == stored.key
    assert lesson.blocks[0].sentences[0].runs[0].term_key == stored.key


def test_conflict_resolution_can_restore_a_gloss_variant_to_stored_identity() -> None:
    prose, units = _two_unit_homograph_fixture()
    stored = _term(
        "stored:hang",
        lemma="行",
        gloss="row",
        pronunciation="háng",
        frequency_rank=321,
    )
    generated = _term(
        "local-1",
        lemma="行",
        gloss="line; queue",
        pronunciation="háng",
        frequency_rank=999,
    )
    results = tuple(_unit_result(unit, generated) for unit in units)
    groups = discover_lexical_conflicts(
        units,
        results,
        learning_language="zh-Hans",
        stored_terms=(stored,),
    )
    generated_candidate = next(
        candidate for candidate in groups[0].candidates if not candidate.stored
    )
    stored_candidate = next(candidate for candidate in groups[0].candidates if candidate.stored)
    resolution = GenerationLexicalConflictResult(
        schema_version=1,
        resolutions=[
            {
                "candidate_id": generated_candidate.candidate_id,
                "canonical_candidate_id": stored_candidate.candidate_id,
            }
        ],
    )

    merged = merge_lexical_unit_results(
        prose,
        results,
        task_id=91,
        learning_language="zh-Hans",
        stored_terms=(stored,),
        conflict_result=resolution,
    )

    assert merged.lessons[0].terms == [stored]


def test_homograph_conflicts_require_resolution_and_conservative_choice_keeps_separate() -> None:
    prose, units = _two_unit_homograph_fixture()
    results = (
        _unit_result(
            units[0],
            _term("noun-local", lemma="行", gloss="row", pronunciation="háng"),
        ),
        _unit_result(
            units[1],
            _term(
                "conduct-local",
                lemma="行",
                gloss="conduct",
                pronunciation="xíng",
            ),
        ),
    )
    stored = _term(
        "stored:industry",
        lemma="行",
        gloss="industry",
        pronunciation="háng",
    )
    groups = discover_lexical_conflicts(
        units,
        results,
        learning_language="zh-Hans",
        stored_terms=(stored,),
    )
    assert len(groups) == 1
    assert len(groups[0].candidates) == 3
    assert sum(candidate.stored for candidate in groups[0].candidates) == 1

    with pytest.raises(ValueError, match="conflict|resolution"):
        merge_lexical_unit_results(
            prose,
            results,
            task_id=10,
            learning_language="zh-Hans",
            stored_terms=(stored,),
        )

    conservative = GenerationLexicalConflictResult(
        schema_version=1,
        resolutions=[
            {
                "candidate_id": candidate.candidate_id,
                "canonical_candidate_id": candidate.candidate_id,
            }
            for candidate in groups[0].candidates
            if not candidate.stored
        ],
    )
    resolved = merge_lexical_unit_results(
        prose,
        results,
        task_id=10,
        learning_language="zh-Hans",
        stored_terms=(stored,),
        conflict_result=conservative,
    )
    resolved_keys = {
        resolved.lessons[0].title_sentence.runs[0].term_key,
        resolved.lessons[0].blocks[0].sentences[0].runs[0].term_key,
    }
    assert None not in resolved_keys
    assert len(resolved_keys) == 2


def test_conflict_results_from_independent_chunks_merge_globally() -> None:
    prose = _prose_result(
        title="行能",
        blocks=(("body", (("body-sentence", "行能。"),)),),
    )
    units = enumerate_lexical_units(prose, "zh-Hans")
    results = (
        GenerationLexicalUnitResult.model_validate(
            {
                "schema_version": 1,
                "unit_id": units[0].unit_id,
                "key": units[0].sentence.key,
                "terms": [
                    _term("row", lemma="行", gloss="row", pronunciation="háng"),
                    _term(
                        "can",
                        lemma="能",
                        gloss="can",
                        pronunciation="néng",
                        pos="verb",
                    ),
                ],
                "runs": [
                    {"text": "行", "term_key": "row", "pronunciation": None},
                    {"text": "能", "term_key": "can", "pronunciation": None},
                ],
            }
        ),
        GenerationLexicalUnitResult.model_validate(
            {
                "schema_version": 1,
                "unit_id": units[1].unit_id,
                "key": units[1].sentence.key,
                "terms": [
                    _term(
                        "walk",
                        lemma="行",
                        gloss="to walk",
                        pronunciation="xíng",
                        pos="verb",
                    ),
                    _term(
                        "ability",
                        lemma="能",
                        gloss="ability",
                        pronunciation="néng",
                    ),
                ],
                "runs": [
                    {"text": "行", "term_key": "walk", "pronunciation": None},
                    {"text": "能", "term_key": "ability", "pronunciation": None},
                    {"text": "。", "term_key": None, "pronunciation": None},
                ],
            }
        ),
    )
    groups = discover_lexical_conflicts(
        units,
        results,
        learning_language="zh-Hans",
        stored_terms=(),
    )
    assert len(groups) == 2
    chunk_results = tuple(
        GenerationLexicalConflictResult(
            schema_version=1,
            resolutions=[
                {
                    "candidate_id": candidate.candidate_id,
                    "canonical_candidate_id": candidate.candidate_id,
                }
                for candidate in group.candidates
                if not candidate.stored
            ],
        )
        for group in groups
    )

    merged = merge_lexical_unit_results(
        prose,
        results,
        task_id=12,
        learning_language="zh-Hans",
        stored_terms=(),
        conflict_results=chunk_results,
    )

    assert len(merged.lessons[0].terms) == 4
    assert "".join(run.text for run in merged.lessons[0].title_sentence.runs) == "行能"
    assert "".join(run.text for run in merged.lessons[0].blocks[0].sentences[0].runs) == "行能。"


def test_merge_matches_unordered_results_and_restores_original_block_order() -> None:
    prose = _prose_result()
    units = enumerate_lexical_units(prose, "zh-Hans")
    results = [
        _unit_result(
            unit,
            _term(
                f"local-{index}",
                lemma=unit.sentence.text.removesuffix("。"),
                gloss=f"term {index}",
                pronunciation=f"reading-{index}",
            ),
        )
        for index, unit in enumerate(units)
    ]

    merged = merge_lexical_unit_results(
        prose,
        tuple(reversed(results)),
        task_id=11,
        learning_language="zh-Hans",
        stored_terms=(),
    )

    lesson = merged.lessons[0]
    assert lesson.title_sentence.key == "title"
    assert [block.key for block in lesson.blocks] == ["block-a", "block-b"]
    assert [[sentence.key for sentence in block.sentences] for block in lesson.blocks] == [
        ["sentence-a", "sentence-b"],
        ["sentence-c"],
    ]
    assert [
        "".join(run.text for run in sentence.runs)
        for block in lesson.blocks
        for sentence in block.sentences
    ] == ["甲。", "乙。", "丙。"]


def _conflicts_for(generated: LessonTerm, stored: LessonTerm) -> list[Any]:
    prose, units = _two_unit_homograph_fixture()
    results = tuple(_unit_result(unit, generated) for unit in units)
    return discover_lexical_conflicts(
        units, results, learning_language="zh-Hans", stored_terms=(stored,)
    )


def test_a_richer_gloss_with_the_same_pos_and_reading_merges_without_the_model() -> None:
    prose, units = _two_unit_homograph_fixture()
    stored = _term("stored:door", lemma="行", gloss="door", pronunciation="háng")
    generated = _term("local-1", lemma="行", gloss="door; gate", pronunciation="háng")
    results = tuple(_unit_result(unit, generated) for unit in units)

    assert _conflicts_for(generated, stored) == []
    merged = merge_lexical_unit_results(
        prose, results, task_id=3, learning_language="zh-Hans", stored_terms=(stored,)
    )
    assert merged.lessons[0].terms == [stored]
    assert merged.lessons[0].title_sentence.runs[0].term_key == stored.key


def test_gloss_order_articles_and_to_do_not_make_a_new_sense() -> None:
    stored = _term(
        "stored:out",
        lemma="行",
        gloss="to go out; to leave home",
        pronunciation="chū háng",
        pos="verb",
    )
    generated = _term(
        "local-1", lemma="行", gloss="To go out; leave home", pronunciation="chū háng", pos="verb"
    )
    assert _conflicts_for(generated, stored) == []


@pytest.mark.parametrize(
    ("pos", "gloss", "pronunciation"),
    [
        ("verb", "door; gate", "háng"),  # a different part of speech
        ("noun", "door; gate", "xíng"),  # a different reading
        ("noun", "gate keeper", "háng"),  # glosses that do not contain one another
    ],
)
def test_ambiguous_variants_still_go_to_model_reconciliation(
    pos: str, gloss: str, pronunciation: str
) -> None:
    stored = _term("stored:door", lemma="行", gloss="door", pronunciation="háng")
    generated = _term("local-1", lemma="行", gloss=gloss, pronunciation=pronunciation, pos=pos)
    groups = _conflicts_for(generated, stored)
    assert len(groups) == 1 and len(groups[0].candidates) == 2


def test_an_anchor_merged_by_the_model_does_not_leave_a_chain() -> None:
    prose, units = _two_unit_homograph_fixture()
    stored = _term("stored:door", lemma="行", gloss="entrance", pronunciation="háng")
    generated_a = _term("local-a", lemma="行", gloss="gate; door", pronunciation="háng")
    generated_b = _term("local-b", lemma="行", gloss="door", pronunciation="háng")
    results = (_unit_result(units[0], generated_a), _unit_result(units[1], generated_b))

    groups = discover_lexical_conflicts(
        units, results, learning_language="zh-Hans", stored_terms=(stored,)
    )
    assert len(groups) == 1
    ids = {candidate.stored: candidate.candidate_id for candidate in groups[0].candidates}
    generated_id = next(c.candidate_id for c in groups[0].candidates if not c.stored)
    resolution = GenerationLexicalConflictResult(
        schema_version=1,
        resolutions=[{"candidate_id": generated_id, "canonical_candidate_id": ids[True]}],
    )
    merged = merge_lexical_unit_results(
        prose,
        results,
        task_id=4,
        learning_language="zh-Hans",
        stored_terms=(stored,),
        conflict_result=resolution,
    )
    assert merged.lessons[0].terms == [stored]
