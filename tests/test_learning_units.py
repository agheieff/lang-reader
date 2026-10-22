from dataclasses import dataclass

import pytest

from server.learning_units import (
    build_learning_unit_index,
    generated_unit_errors,
    is_learning_unit,
    learning_unit_issue,
    term_identity,
)
from server.schemas import (
    GeneratedLessonBlock,
    GeneratedLessonDraft,
    LessonBlock,
    LessonDocument,
    LessonRun,
    LessonSentence,
    LessonTerm,
)


@dataclass(frozen=True)
class Term:
    key: str
    lemma: str
    pos: str
    gloss: str = "test"
    pronunciation: str | None = None


@pytest.mark.parametrize(
    ("lemma", "pos"),
    [
        ("一个", "numeral phrase"),
        ("一个", "noun"),
        ("两个", "quantifier"),
        ("几张", "quantifier"),
        ("几张", "noun"),
        ("第二份", "quantifier"),
        ("第二份", "ordinal"),
        ("一周", "noun"),
        ("一周", "phrase"),
        ("三天", "noun"),
        ("三天", "phrase"),
        ("每个", "determiner"),
        ("每个人", "pronoun"),
        ("这个", "determiner"),
        ("每样", "pronoun"),
        ("这次", "noun"),
        ("笑了", "verb"),
        ("想着", "verb"),
        ("有什么", "phrase"),
        ("四十五分", "noun"),
        ("看错", "verb"),
        ("听到", "verb"),
        ("关掉", "verb"),
        ("拿出来", "verb"),
    ],
)
def test_productive_chinese_expressions_are_not_learning_units(lemma: str, pos: str) -> None:
    term = Term(f"key-{lemma}", lemma, pos)

    assert not is_learning_unit("zh-Hans", term)
    assert learning_unit_issue("zh-Hans", term) is not None


@pytest.mark.parametrize(
    ("lemma", "pos"),
    [
        ("一", "numeral"),
        ("个", "classifier"),
        ("一些", "determiner"),
        ("一起", "adverb"),
        ("一直", "adverb"),
        ("一生", "noun"),
        ("一块", "adverb"),
        ("一样", "adjective"),
        ("一樣", "adverb"),
        ("这样", "pronoun"),
        ("这样", "adverb"),
        ("這樣", "pronoun"),
        ("那样", "pronoun"),
        ("那樣", "pronoun"),
        ("每天", "noun"),
        ("下一步", "noun"),
        ("上个月", "noun"),
        ("四十", "numeral"),
        ("十几", "numeral"),
        ("为了", "preposition"),
        ("除了", "preposition"),
        ("得到", "verb"),
        ("迟到", "verb"),
        ("感到", "verb"),
        ("知道", "verb"),
        ("出来", "verb"),
        ("起来", "verb"),
        ("觉得", "verb"),
    ],
)
def test_lexicalized_chinese_units_survive_conservative_policy(lemma: str, pos: str) -> None:
    assert is_learning_unit("zh-Hans", Term(f"key-{lemma}", lemma, pos))


def test_other_languages_are_not_stemmed_or_split() -> None:
    assert is_learning_unit("de-DE", Term("de", "Staubsauger", "noun"))
    assert is_learning_unit("es-ES", Term("es", "al", "contraction"))


def test_chinese_same_word_exception_remains_sense_specific() -> None:
    assert is_learning_unit("zh-Hans", Term("same", "一样", "adjective", "the same"))
    assert not is_learning_unit("zh-Hans", Term("one-kind", "一样", "noun", "one kind"))


def test_exact_identity_ignores_frequency_but_not_meaning() -> None:
    first = Term("first", "了", "particle", "completed-action marker", "le")
    alias = Term("alias", "了", " particle ", "completed-action marker", "le")
    other = Term("other", "了", "particle", "change-of-state marker", "le")

    assert term_identity(first) == term_identity(alias)
    assert term_identity(first) != term_identity(other)


def test_chinese_identity_ignores_pinyin_spacing_but_not_the_reading() -> None:
    compact = Term("compact", "志愿者", "noun", "volunteer", "zhìyuànzhě")
    spaced = Term("spaced", "志愿者", "noun", "volunteer", "zhì yuàn zhě")
    other_reading = Term("other", "志愿者", "noun", "volunteer", "zhìyuánzhě")

    assert term_identity(compact, "zh-Hans") == term_identity(spaced, "zh-Hans")
    assert term_identity(compact, "zh-Hans") != term_identity(other_reading, "zh-Hans")
    assert term_identity(compact, "und") != term_identity(spaced, "und")


def test_index_aliases_exact_identity_and_ignores_pinyin_spacing() -> None:
    compact = LessonTerm(
        key="compact", lemma="志愿者", pos="noun", gloss="volunteer", pronunciation="zhìyuànzhě"
    )
    spaced = compact.model_copy(update={"key": "spaced", "pronunciation": "zhì yuàn zhě"})
    other_reading = compact.model_copy(update={"key": "other", "pronunciation": "zhìyuánzhě"})

    index = build_learning_unit_index(
        [lesson_with(compact), lesson_with(spaced), lesson_with(other_reading)]
    )

    assert index.resolve("compact") == "compact"
    assert index.resolve("spaced") == "compact"
    assert index.resolve("other") == "other"


def test_index_rejects_one_key_with_two_definitions() -> None:
    early = LessonTerm(key="same", lemma="早", pos="noun", gloss="early")
    late = LessonTerm(key="same", lemma="晚", pos="noun", gloss="late")

    with pytest.raises(ValueError, match="conflicting definitions"):
        build_learning_unit_index([lesson_with(early), lesson_with(late)])


def lesson_with(term: LessonTerm, *, text: str | None = None) -> LessonDocument:
    return LessonDocument(
        key=f"lesson-{term.key}",
        title="测试",
        learning_language="zh-Hans",
        translation_language="en",
        blocks=[
            LessonBlock(
                key="b1",
                sentences=[
                    LessonSentence(
                        key="s1",
                        runs=[LessonRun(text=text or term.lemma, term=term)],
                    )
                ],
            )
        ],
    )


def test_index_keeps_display_terms_but_resolves_only_canonical_learning_units() -> None:
    canonical = LessonTerm(
        key="le", lemma="了", pos="particle", gloss="completed", pronunciation="le"
    )
    alias = canonical.model_copy(update={"key": "le-alias", "frequency_rank": 10})
    composite = LessonTerm(key="two-cl", lemma="两个", pos="quantifier", gloss="two")

    index = build_learning_unit_index(
        [lesson_with(canonical), lesson_with(alias), lesson_with(composite)]
    )

    assert index.resolve("le") == "le"
    assert index.resolve("le-alias") == "le"
    assert index.resolve("two-cl") is None
    assert list(index.learning_catalog()) == ["le"]


def test_index_exposes_ordered_component_candidates_without_guessing_meaning() -> None:
    laugh = LessonTerm(key="laugh", lemma="笑", pos="verb", gloss="laugh")
    completed = LessonTerm(key="le-completed", lemma="了", pos="particle", gloss="completed action")
    changed = LessonTerm(key="le-changed", lemma="了", pos="particle", gloss="change of state")
    composite = LessonTerm(key="laughed", lemma="笑了", pos="verb", gloss="laughed")

    index = build_learning_unit_index(
        [
            lesson_with(laugh),
            lesson_with(completed),
            lesson_with(changed),
            lesson_with(composite),
        ]
    )

    decomposition = index.decomposition("laughed")
    assert decomposition is not None
    assert [component.surface for component in decomposition.components] == ["笑", "了"]
    candidates = index.component_candidates("laughed")
    assert [item.component.role for item in candidates] == ["lexical-base", "aspect-marker"]
    assert candidates[0].canonical_keys == ("laugh",)
    assert candidates[1].canonical_keys == ("le-completed", "le-changed")
    assert index.resolve("laughed") is None
    assert index.select_evidence_targets("laughed", {}) == ("laugh", "le-changed")
    assert index.select_evidence_targets("laughed", {"le-completed": 1.0}) == (
        "laugh",
        "le-completed",
    )


def test_index_resolves_unambiguous_component_evidence_without_fabricating_terms() -> None:
    have = LessonTerm(key="have", lemma="有", pos="verb", gloss="have")
    what = LessonTerm(key="what", lemma="什么", pos="pronoun", gloss="what")
    composite = LessonTerm(key="have-what", lemma="有什么", pos="phrase", gloss="have anything")

    index = build_learning_unit_index(
        [lesson_with(have), lesson_with(what), lesson_with(composite)]
    )

    assert index.select_evidence_targets("have-what", {}) == ("have", "what")
    assert index.resolve("have-what") is None
    assert index.component_candidates("missing") == ()


def test_index_projects_a_known_chinese_subject_predicate_span_to_words() -> None:
    floor = LessonTerm(key="floor", lemma="地板", pos="noun", gloss="floor")
    vibrate = LessonTerm(key="vibrate", lemma="震动", pos="verb", gloss="vibrate")
    phrase = LessonTerm(
        key="floor-vibration",
        lemma="地板震动",
        pos="noun",
        gloss="vibration of the floor",
    )

    index = build_learning_unit_index(
        [lesson_with(floor), lesson_with(vibrate), lesson_with(phrase)]
    )

    assert index.resolve("floor-vibration") is None
    assert index.select_evidence_targets("floor-vibration", {}) == ("floor", "vibrate")


def test_index_keeps_unresolved_component_slots_visible() -> None:
    composite = LessonTerm(
        key="forty-five-minutes",
        lemma="四十五分",
        pos="noun",
        gloss="forty-five minutes",
    )

    index = build_learning_unit_index([lesson_with(composite)])

    candidates = index.component_candidates("forty-five-minutes")
    assert [item.component.surface for item in candidates] == ["四十五", "分"]
    assert [item.canonical_keys for item in candidates] == [(), ()]
    assert index.select_evidence_targets("forty-five-minutes", {}) == ()


def test_generated_validator_accepts_known_composite_and_adjacent_units() -> None:
    one = LessonTerm(key="one", lemma="一", pos="numeral", gloss="one", frequency_rank=2)
    classifier = LessonTerm(
        key="classifier", lemma="个", pos="classifier", gloss="general classifier", frequency_rank=3
    )
    composite = LessonTerm(
        key="one-classifier",
        lemma="一个",
        pos="numeral phrase",
        gloss="one; a",
        frequency_rank=4,
    )

    def draft(runs: list[LessonRun]) -> GeneratedLessonDraft:
        title_term = LessonTerm(
            key="test", lemma="测试", pos="noun", gloss="test", frequency_rank=100
        )
        title_sentence = LessonSentence(
            key="title", runs=[LessonRun(text="测试", term=title_term)], translation="Test"
        )
        return GeneratedLessonDraft(
            title="测试",
            title_sentence=title_sentence,
            topic="test",
            level="A1",
            difficulty=0.1,
            blocks=[
                GeneratedLessonBlock(
                    key="b1",
                    sentences=[LessonSentence(key="s1", runs=runs, translation="One item")],
                )
            ],
            target_term_keys=[],
        )

    assert not generated_unit_errors(
        draft([LessonRun(text="一个", term=composite)]),
        "zh-Hans",
    )
    assert not generated_unit_errors(
        draft([LessonRun(text="一", term=one), LessonRun(text="个", term=classifier)]),
        "zh-Hans",
    )


def test_generated_validator_treats_configured_chinese_split_as_recoverable() -> None:
    laughed = LessonTerm(
        key="laughed",
        lemma="笑了",
        pos="verb",
        gloss="laughed",
        frequency_rank=500,
    )
    title = LessonTerm(key="test", lemma="测试", pos="noun", gloss="test", frequency_rank=100)
    draft = GeneratedLessonDraft(
        title="测试",
        title_sentence=LessonSentence(
            key="title",
            runs=[LessonRun(text="测试", term=title)],
            translation="Test",
        ),
        topic="test",
        level="A1",
        difficulty=0.1,
        blocks=[
            GeneratedLessonBlock(
                key="b1",
                sentences=[
                    LessonSentence(
                        key="s1",
                        runs=[LessonRun(text="笑了", term=laughed)],
                        translation="Laughed.",
                    )
                ],
            )
        ],
        target_term_keys=[],
    )

    errors = generated_unit_errors(draft, "zh-Hans")

    assert errors == []


def test_generated_validator_keeps_lexicalized_zheyang_but_splits_productive_forms() -> None:
    zheyang = LessonTerm(
        key="zheyang",
        lemma="这样",
        pos="pronoun",
        gloss="this way; like this",
        frequency_rank=250,
    )
    every_kind = LessonTerm(
        key="every-kind",
        lemma="每样",
        pos="pronoun",
        gloss="every kind",
        frequency_rank=900,
    )
    this = LessonTerm(
        key="this",
        lemma="这",
        pos="pronoun",
        gloss="this",
        frequency_rank=30,
    )

    def draft(run: LessonRun) -> GeneratedLessonDraft:
        title_term = LessonTerm(
            key="test", lemma="测试", pos="noun", gloss="test", frequency_rank=100
        )
        return GeneratedLessonDraft(
            title="测试",
            title_sentence=LessonSentence(
                key="title",
                runs=[LessonRun(text="测试", term=title_term)],
                translation="Test",
            ),
            topic="test",
            level="A1",
            difficulty=0.1,
            blocks=[
                GeneratedLessonBlock(
                    key="b1",
                    sentences=[LessonSentence(key="s1", runs=[run], translation="Test")],
                )
            ],
            target_term_keys=[],
        )

    assert not generated_unit_errors(draft(LessonRun(text="这样", term=zheyang)), "zh-Hans")
    every_kind_errors = generated_unit_errors(
        draft(LessonRun(text="每样", term=every_kind)), "zh-Hans"
    )
    mismatched_surface_errors = generated_unit_errors(
        draft(LessonRun(text="这个", term=this)), "zh-Hans"
    )

    assert every_kind_errors == []
    assert any("surface text must equal" in error for error in mismatched_surface_errors)
