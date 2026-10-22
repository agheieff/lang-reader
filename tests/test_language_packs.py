from pathlib import Path

import pytest
from pydantic import ValidationError

from server.language_packs import (
    GENERIC_LANGUAGE_PACK,
    LearningComponent,
    language_pack,
    language_pack_digest,
    language_packs,
    load_language_packs,
    resolve_language_pack,
)


def test_tracked_language_packs_load_and_route_variants() -> None:
    packs = {pack.id: pack for pack in language_packs()}

    assert set(packs) == {"de", "es", "zh"}
    assert language_pack("zh-Hant").id == "zh"
    assert language_pack("zh-Hant").pronunciation_identity_ignores_whitespace is True
    assert language_pack("zh-Hans").normalize_pronunciation("zhì yuàn  zhě") == "zhìyuànzhě"
    assert any("这次 -> 这 + 次" in item for item in language_pack("zh-Hans").generation_guidance)
    assert any(
        "第一次 -> 第 + 一 + 次" in item for item in language_pack("zh-Hans").generation_guidance
    )
    assert any("这种 -> 这 + 种" in item for item in language_pack("zh-Hans").generation_guidance)
    assert any("surface text 这种" in item for item in language_pack("zh-Hans").generation_guidance)
    assert any("地板 + 震动" in item for item in language_pack("zh-Hans").generation_guidance)
    assert language_pack("es-419").normalize_pos("verbo") == "verb"
    assert language_pack("de-DE").normalize_pos("Nomen") == "noun"
    assert language_pack("fi-FI") == GENERIC_LANGUAGE_PACK
    assert len(language_pack_digest("zh-Hans")) == 64


def test_chinese_contextual_pronunciations_are_declarative() -> None:
    pack = language_pack("zh-Hans")

    assert (
        pack.contextual_pronunciation(
            lemma="一",
            pos="numeral",
            canonical_pronunciation="yī",
            following_pronunciation="gè",
        )
        == "yí"
    )
    assert (
        pack.contextual_pronunciation(
            lemma="一",
            pos="adverb",
            canonical_pronunciation="yī",
            following_pronunciation="shǔ",
        )
        == "yì"
    )
    assert (
        pack.contextual_pronunciation(
            lemma="不",
            pos="adverb",
            canonical_pronunciation="bù",
            following_pronunciation="kàn",
        )
        == "bú"
    )
    assert (
        pack.contextual_pronunciation(
            lemma="不",
            pos="adverb",
            canonical_pronunciation="bù",
            following_pronunciation="hǎo",
        )
        == "bù"
    )
    assert (
        pack.contextual_pronunciation(
            lemma="行",
            pos="verb",
            canonical_pronunciation="háng",
            following_pronunciation="kàn",
        )
        is None
    )


def test_base_and_variant_packs_merge_in_specificity_order(tmp_path: Path) -> None:
    (tmp_path / "es.toml").write_text(
        """
schema_version = 1
id = "es"
tags = ["es"]
description = "base"
generation_guidance = ["base guidance"]
[pos_aliases]
v = "verb"
[[non_learning_patterns]]
id = "demo"
lemma_regex = '^base$'
reason = "base reason"
[[decomposition_patterns]]
id = "suffix"
lemma_regex = '^(?P<base>.+)(?P<suffix>x)$'
reason = "base decomposition"
[[decomposition_patterns.components]]
group = "base"
role = "base"
[[decomposition_patterns.components]]
group = "suffix"
role = "suffix"
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "es-419.toml").write_text(
        """
schema_version = 1
id = "es-419"
tags = ["es-419"]
description = "variant"
generation_guidance = ["variant guidance"]
[[non_learning_patterns]]
id = "demo"
lemma_regex = '^variant$'
reason = "variant reason"
[[decomposition_patterns]]
id = "suffix"
lemma_regex = '^(?P<base>.+)(?P<suffix>y)$'
reason = "variant decomposition"
[[decomposition_patterns.components]]
group = "base"
role = "base"
[[decomposition_patterns.components]]
group = "suffix"
role = "suffix"
""".strip(),
        encoding="utf-8",
    )

    merged = resolve_language_pack("es-419", load_language_packs(sorted(tmp_path.glob("*.toml"))))

    assert merged.description == "variant"
    assert merged.generation_guidance == ("base guidance", "variant guidance")
    assert merged.normalize_pos("v") == "verb"
    assert merged.learning_issue("base", "noun") is None
    assert merged.learning_issue("variant", "noun") == "variant reason"
    assert merged.decompose("play", "verb") is not None
    assert merged.decompose("play", "verb").reason == "variant decomposition"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("lemma", "pos", "rule_id", "components"),
    [
        (
            "笑了",
            "verb",
            "verb-aspect-suffix",
            (
                LearningComponent("笑", "lexical-base", ("verb",)),
                LearningComponent("了", "aspect-marker", ("particle",)),
            ),
        ),
        (
            "想着",
            "verb",
            "verb-aspect-suffix",
            (
                LearningComponent("想", "lexical-base", ("verb",)),
                LearningComponent("着", "aspect-marker", ("particle",)),
            ),
        ),
        (
            "有什么",
            "phrase",
            "existential-question",
            (
                LearningComponent("有", "lexical-base", ("verb",)),
                LearningComponent("什么", "interrogative-pronoun", ("pronoun",)),
            ),
        ),
        (
            "四十五分",
            "noun",
            "number-measure",
            (
                LearningComponent("四十五", "number", ("numeral",)),
                LearningComponent("分", "measure", ("classifier", "noun", "measure")),
            ),
        ),
        (
            "地板震动",
            "noun",
            "floor-vibration-clause",
            (
                LearningComponent("地板", "subject", ("noun",)),
                LearningComponent("震动", "predicate", ("verb",)),
            ),
        ),
        (
            "几道",
            "quantifier",
            "number-classifier",
            (
                LearningComponent("几", "number", ("numeral",)),
                LearningComponent("道", "classifier", ("classifier", "noun")),
            ),
        ),
        (
            "一户",
            "quantifier",
            "number-classifier",
            (
                LearningComponent("一", "number", ("numeral",)),
                LearningComponent("户", "classifier", ("classifier", "noun")),
            ),
        ),
        (
            "另一个",
            "determiner",
            "compound-determiner-classifier",
            (
                LearningComponent("另", "determiner", ("determiner", "pronoun")),
                LearningComponent("一", "number", ("numeral",)),
                LearningComponent("个", "classifier", ("classifier",)),
            ),
        ),
        (
            "听到",
            "verb",
            "attainment-complement",
            (
                LearningComponent("听", "lexical-base", ("verb",)),
                LearningComponent("到", "result-complement", ("complement", "verb")),
            ),
        ),
        (
            "不要",
            "auxiliary",
            "negation-modal-yao",
            (
                LearningComponent("不", "negator", ("adverb", "particle")),
                LearningComponent("要", "modal", ("auxiliary", "modal", "verb")),
            ),
        ),
        (
            "不是",
            "verb",
            "negation-copula-shi",
            (
                LearningComponent("不", "negator", ("adverb", "particle")),
                LearningComponent("是", "copula", ("auxiliary", "verb")),
            ),
        ),
        (
            "拿出来",
            "verb",
            "directional-complement",
            (
                LearningComponent("拿", "lexical-base", ("verb",)),
                LearningComponent("出来", "directional-complement", ("complement", "verb")),
            ),
        ),
        (
            "拿出来了",
            "verb",
            "verb-aspect-suffix",
            (
                LearningComponent("拿", "lexical-base", ("verb",)),
                LearningComponent("出来", "directional-complement", ("complement", "verb")),
                LearningComponent("了", "aspect-marker", ("particle",)),
            ),
        ),
    ],
)
def test_chinese_decomposition_is_ordered_and_declarative(
    lemma: str,
    pos: str,
    rule_id: str,
    components: tuple[LearningComponent, ...],
) -> None:
    decomposition = language_pack("zh-Hans").decompose(lemma, pos)

    assert decomposition is not None
    assert decomposition.rule_id == rule_id
    assert decomposition.components == components


@pytest.mark.parametrize(
    ("lemma", "pos"),
    [
        ("为了", "preposition"),
        ("除了", "preposition"),
        ("罢了", "particle"),
        ("得到", "verb"),
        ("迟到", "verb"),
        ("感到", "verb"),
        ("知道", "verb"),
        ("出来", "verb"),
        ("起来", "verb"),
        ("觉得", "verb"),
        ("不要紧", "adjective"),
        ("不得不", "auxiliary"),
    ],
)
def test_chinese_decomposition_preserves_lexicalized_entries(lemma: str, pos: str) -> None:
    assert language_pack("zh-Hans").decompose(lemma, pos) is None


def test_invalid_pack_fails_fast(tmp_path: Path) -> None:
    (tmp_path / "broken.toml").write_text(
        """
schema_version = 1
id = "broken"
tags = ["xx"]
description = "broken"
unknown = true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_language_packs(sorted(tmp_path.glob("*.toml")))


def test_contextual_pronunciation_pack_rejects_unknown_classes(tmp_path: Path) -> None:
    (tmp_path / "broken.toml").write_text(
        """
schema_version = 1
id = "broken"
tags = ["xx"]
description = "broken"
[[pronunciation_classes]]
id = "known"
first_unit_regex = '^x'
[[contextual_pronunciation_rules]]
id = "bad-context"
lemma = "x"
canonical_pronunciations = ["x"]
following_pronunciations = { missing = "y" }
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="references unknown classes"):
        load_language_packs(sorted(tmp_path.glob("*.toml")))


def test_decomposition_pack_fails_when_a_component_capture_is_missing(tmp_path: Path) -> None:
    (tmp_path / "broken.toml").write_text(
        """
schema_version = 1
id = "broken"
tags = ["xx"]
description = "broken"
[[decomposition_patterns]]
id = "bad-split"
lemma_regex = '^(?P<base>.+)x$'
reason = "bad"
[[decomposition_patterns.components]]
group = "base"
role = "base"
[[decomposition_patterns.components]]
group = "missing"
role = "suffix"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="missing named groups"):
        load_language_packs(sorted(tmp_path.glob("*.toml")))
