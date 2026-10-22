"""Declarative language conventions loaded from tracked TOML rule packs."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[import-not-found]

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from server.packs import pack_files


class LanguagePattern(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=80)
    lemma_regex: str = Field(min_length=1, max_length=2_000)
    pos: tuple[str, ...] = ()
    reason: str = Field(min_length=1, max_length=300)
    examples: tuple[str, ...] = ()

    @field_validator("lemma_regex")
    @classmethod
    def valid_regex(cls, value: str) -> str:
        re.compile(value)
        return value


class LemmaConvention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=80)
    pos: tuple[str, ...] = ()
    guidance: str = Field(min_length=1, max_length=1_000)
    lemma_regex: str | None = Field(default=None, max_length=2_000)
    severity: Literal["guidance", "error"] = "guidance"

    @field_validator("lemma_regex")
    @classmethod
    def valid_optional_regex(cls, value: str | None) -> str | None:
        if value is not None:
            re.compile(value)
        return value


class LearningOverride(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lemma: str = Field(min_length=1, max_length=200)
    pos: tuple[str, ...] = ()
    decision: Literal["include", "exclude"]
    reason: str = Field(min_length=1, max_length=300)


class DecompositionComponentSpec(BaseModel):
    """One ordered capture from a productive-expression rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    role: str = Field(min_length=1, max_length=80)
    pos: tuple[str, ...] = ()


class DecompositionPattern(BaseModel):
    """A deterministic split whose captures remain display text, not invented definitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=80)
    lemma_regex: str = Field(min_length=1, max_length=2_000)
    pos: tuple[str, ...] = ()
    reason: str = Field(min_length=1, max_length=300)
    examples: tuple[str, ...] = ()
    recompute_contextual_pronunciation: bool = False
    components: tuple[DecompositionComponentSpec, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def valid_captures(self) -> DecompositionPattern:
        expression = re.compile(self.lemma_regex)
        groups = [component.group for component in self.components]
        if len(groups) != len(set(groups)):
            raise ValueError("decomposition component groups must be unique")
        missing = set(groups) - expression.groupindex.keys()
        if missing:
            raise ValueError(f"decomposition regex is missing named groups: {sorted(missing)}")
        for example in self.examples:
            match = expression.fullmatch(example)
            captures = [match.group(group) for group in groups] if match is not None else []
            if not captures or any(not capture for capture in captures):
                raise ValueError(f"decomposition example does not match every component: {example}")
            if "".join(captures) != example:
                raise ValueError(f"decomposition components do not cover the example: {example}")
        return self


class PronunciationClass(BaseModel):
    """A declarative class matched against the first unit of a pronunciation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$", max_length=80)
    first_unit_regex: str = Field(min_length=2, max_length=2_000)

    @field_validator("first_unit_regex")
    @classmethod
    def valid_first_unit_regex(cls, value: str) -> str:
        if not value.startswith("^"):
            raise ValueError("first-unit pronunciation regex must be anchored at the start")
        re.compile(value)
        return value


class ContextualPronunciationRule(BaseModel):
    """A surface reading selected from the following lexical pronunciation class."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$", max_length=80)
    lemma: str = Field(min_length=1, max_length=200)
    pos: tuple[str, ...] = ()
    canonical_pronunciations: tuple[str, ...] = Field(min_length=1)
    following_pronunciations: dict[str, str] = Field(min_length=1)

    @field_validator("lemma")
    @classmethod
    def normalize_lemma(cls, value: str) -> str:
        return unicodedata.normalize("NFC", value.strip())

    @field_validator("canonical_pronunciations")
    @classmethod
    def valid_canonical_pronunciations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(unicodedata.normalize("NFC", value.strip()) for value in values)
        if any(not value for value in normalized):
            raise ValueError("canonical pronunciations must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("canonical pronunciations must be unique")
        return normalized

    @field_validator("following_pronunciations")
    @classmethod
    def valid_following_pronunciations(cls, values: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for pronunciation_class, pronunciation in values.items():
            if re.fullmatch(r"[a-z][a-z0-9_-]*", pronunciation_class) is None:
                raise ValueError(f"invalid pronunciation class id: {pronunciation_class!r}")
            reading = unicodedata.normalize("NFC", pronunciation.strip())
            if not reading:
                raise ValueError("contextual pronunciations must not be blank")
            normalized[pronunciation_class] = reading
        return normalized


@dataclass(frozen=True)
class LearningComponent:
    """A surface component identified by a pack, without a fabricated dictionary entry."""

    surface: str
    role: str
    pos: tuple[str, ...]


@dataclass(frozen=True)
class LearningDecomposition:
    rule_id: str
    reason: str
    components: tuple[LearningComponent, ...]
    recompute_contextual_pronunciation: bool = False


class LanguagePack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    tags: tuple[str, ...] = Field(min_length=1)
    description: str = Field(min_length=1, max_length=300)
    generation_guidance: tuple[str, ...] = ()
    pos_aliases: dict[str, str] = Field(default_factory=dict)
    pronunciation_identity_ignores_whitespace: bool = False
    generated_surface_must_equal_lemma: bool = False
    learning_overrides: tuple[LearningOverride, ...] = ()
    learning_exceptions: tuple[LanguagePattern, ...] = ()
    non_learning_patterns: tuple[LanguagePattern, ...] = ()
    decomposition_patterns: tuple[DecompositionPattern, ...] = ()
    # Per-instance memoization: a pack is immutable and replays query the same terms repeatedly.
    _pos_sets: dict[tuple[str, ...], frozenset[str]] = PrivateAttr(default_factory=dict)
    _issues: dict[tuple[str, str], str | None] = PrivateAttr(default_factory=dict)
    _decompositions: dict[tuple[str, str], Any] = PrivateAttr(default_factory=dict)
    lemma_conventions: tuple[LemmaConvention, ...] = ()
    pronunciation_classes: tuple[PronunciationClass, ...] = ()
    contextual_pronunciation_rules: tuple[ContextualPronunciationRule, ...] = ()

    @model_validator(mode="after")
    def valid_contextual_pronunciation_rules(self) -> LanguagePack:
        class_ids = [item.id for item in self.pronunciation_classes]
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("pronunciation class ids must be unique")
        rule_ids = [item.id for item in self.contextual_pronunciation_rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("contextual pronunciation rule ids must be unique")
        known_classes = set(class_ids)
        for rule in self.contextual_pronunciation_rules:
            unknown = set(rule.following_pronunciations) - known_classes
            if unknown:
                raise ValueError(
                    f"contextual pronunciation rule {rule.id!r} references unknown classes: "
                    f"{sorted(unknown)}"
                )
        return self

    def normalize_pos(self, value: str) -> str:
        normalized = normalize_text(value)
        return self.pos_aliases.get(normalized, normalized)

    def normalize_pronunciation(self, value: str) -> str:
        normalized = normalize_text(value)
        return (
            normalized.replace(" ", "")
            if self.pronunciation_identity_ignores_whitespace
            else normalized
        )

    def contextual_pronunciation(
        self,
        *,
        lemma: str,
        pos: str,
        canonical_pronunciation: str | None,
        following_pronunciation: str,
    ) -> str | None:
        """Return a configured surface reading, without changing the dictionary reading."""

        if canonical_pronunciation is None:
            return None
        normalized_lemma = unicodedata.normalize("NFC", lemma)
        normalized_pos = self.normalize_pos(pos)
        pronunciation_class = self._pronunciation_class(following_pronunciation)
        if pronunciation_class is None:
            return None
        outcomes: dict[str, str] = {}
        normalized_canonical = self.normalize_pronunciation(canonical_pronunciation)
        for rule in self.contextual_pronunciation_rules:
            if (
                normalized_lemma != rule.lemma
                or not _pos_matches(rule.pos, normalized_pos, self)
                or normalized_canonical
                not in {
                    self.normalize_pronunciation(value) for value in rule.canonical_pronunciations
                }
            ):
                continue
            pronunciation = rule.following_pronunciations.get(pronunciation_class)
            if pronunciation is not None:
                outcomes[self.normalize_pronunciation(pronunciation)] = pronunciation
        if len(outcomes) > 1:
            raise ValueError(
                f"contextual pronunciation rules disagree for {lemma!r} before "
                f"{following_pronunciation!r}"
            )
        return next(iter(outcomes.values()), None)

    def _pronunciation_class(self, pronunciation: str) -> str | None:
        normalized = normalize_text(unicodedata.normalize("NFC", pronunciation))
        matches = [
            item.id
            for item in self.pronunciation_classes
            if re.search(item.first_unit_regex, normalized) is not None
        ]
        if len(matches) > 1:
            raise ValueError(f"pronunciation {pronunciation!r} matches multiple classes: {matches}")
        return matches[0] if matches else None

    def normalized_pos_set(self, allowed: tuple[str, ...]) -> frozenset[str]:
        cached = self._pos_sets.get(allowed)
        if cached is None:
            cached = frozenset(self.normalize_pos(item) for item in allowed)
            self._pos_sets[allowed] = cached
        return cached

    def learning_issue(self, lemma: str, pos: str) -> str | None:
        key = (lemma, pos)
        if key not in self._issues:
            self._issues[key] = self._learning_issue(lemma, pos)
        return self._issues[key]

    def _learning_issue(self, lemma: str, pos: str) -> str | None:
        normalized_lemma = unicodedata.normalize("NFC", lemma)
        normalized_pos = self.normalize_pos(pos)
        for override in self.learning_overrides:
            if normalized_lemma == override.lemma and _pos_matches(
                override.pos, normalized_pos, self
            ):
                return None if override.decision == "include" else override.reason
        for pattern in self.learning_exceptions:
            if _pattern_matches(pattern, normalized_lemma, normalized_pos, self):
                return None
        if decomposition := self._decompose(normalized_lemma, normalized_pos, frozenset()):
            return decomposition.reason
        for pattern in self.non_learning_patterns:
            if _pattern_matches(pattern, normalized_lemma, normalized_pos, self):
                return pattern.reason
        return None

    def decompose(self, lemma: str, pos: str) -> LearningDecomposition | None:
        """Return a configured split unless an explicit include/exception preserves the lemma."""

        key = (lemma, pos)
        if key not in self._decompositions:
            normalized_lemma = unicodedata.normalize("NFC", lemma)
            self._decompositions[key] = self._decompose(
                normalized_lemma, self.normalize_pos(pos), frozenset()
            )
        return self._decompositions[key]  # type: ignore[no-any-return]

    def _decompose(
        self, lemma: str, normalized_pos: str, ancestors: frozenset[str]
    ) -> LearningDecomposition | None:
        if lemma in ancestors or self._preserves_learning_unit(lemma, normalized_pos):
            return None
        for pattern in self.decomposition_patterns:
            if not _pos_matches(pattern.pos, normalized_pos, self):
                continue
            match = re.fullmatch(pattern.lemma_regex, lemma)
            if match is None:
                continue
            captures = [match.group(component.group) for component in pattern.components]
            if any(not capture for capture in captures) or "".join(captures) != lemma:
                continue
            children = tuple(
                LearningComponent(capture, component.role, component.pos)
                for capture, component in zip(captures, pattern.components, strict=True)
            )
            return LearningDecomposition(
                rule_id=pattern.id,
                reason=pattern.reason,
                components=self._expand_components(children, ancestors | {lemma}),
                recompute_contextual_pronunciation=pattern.recompute_contextual_pronunciation,
            )
        return None

    def _expand_components(
        self, components: tuple[LearningComponent, ...], ancestors: frozenset[str]
    ) -> tuple[LearningComponent, ...]:
        expanded: list[LearningComponent] = []
        for component in components:
            nested = {
                decomposition
                for pos in component.pos
                if (
                    decomposition := self._decompose(
                        component.surface, self.normalize_pos(pos), ancestors
                    )
                )
                is not None
            }
            if len(nested) == 1:
                expanded.extend(nested.pop().components)
            else:
                expanded.append(component)
        return tuple(expanded)

    def _preserves_learning_unit(self, lemma: str, normalized_pos: str) -> bool:
        return any(
            lemma == override.lemma and _pos_matches(override.pos, normalized_pos, self)
            for override in self.learning_overrides
        ) or any(
            _pattern_matches(pattern, lemma, normalized_pos, self)
            for pattern in self.learning_exceptions
        )

    def lemma_errors(self, lemma: str, pos: str) -> list[str]:
        normalized_pos = self.normalize_pos(pos)
        return [
            convention.guidance
            for convention in self.lemma_conventions
            if convention.severity == "error"
            and _pos_matches(convention.pos, normalized_pos, self)
            and convention.lemma_regex is not None
            and re.fullmatch(convention.lemma_regex, lemma) is None
        ]

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()


GENERIC_LANGUAGE_PACK = LanguagePack(
    schema_version=1,
    id="generic",
    tags=("und",),
    description="Safe defaults for a language without a specialized rule pack.",
    generation_guidance=(
        "Use dictionary headwords as lemmas and reuse one stable key for the same lemma, part of "
        "speech, and meaning; keep genuine homographs and meanings separate.",
        "Annotate reusable lexical units rather than arbitrary multiword spans.",
    ),
)


def language_pack(language_tag: str) -> LanguagePack:
    return _cached_language_pack(normalize_language_tag(language_tag), _rules_signature())


@lru_cache(maxsize=128)
def _cached_language_pack(
    language_tag: str, _signature: tuple[tuple[str, int, int], ...]
) -> LanguagePack:
    return resolve_language_pack(language_tag, language_packs())


def resolve_language_pack(language_tag: str, packs: tuple[LanguagePack, ...]) -> LanguagePack:
    normalized = normalize_language_tag(language_tag)
    matches = [
        (
            max(
                len(tag)
                for tag in pack.tags
                if _tag_matches(normalized, normalize_language_tag(tag))
            ),
            pack,
        )
        for pack in packs
        if any(_tag_matches(normalized, normalize_language_tag(tag)) for tag in pack.tags)
    ]
    if not matches:
        return GENERIC_LANGUAGE_PACK
    ordered = [pack for _, pack in sorted(matches, key=lambda item: item[0])]
    return _merge_language_packs(normalized, ordered)


def language_packs() -> tuple[LanguagePack, ...]:
    return _cached_language_packs(_rules_signature())


@lru_cache(maxsize=16)
def _cached_language_packs(
    _signature: tuple[tuple[str, int, int], ...],
) -> tuple[LanguagePack, ...]:
    return load_language_packs(pack_files("rules"))


def load_language_packs(paths: Iterable[Path]) -> tuple[LanguagePack, ...]:
    packs: list[LanguagePack] = []
    seen_ids: set[str] = set()
    seen_tags: set[str] = set()
    for path in paths:
        with path.open("rb") as stream:
            raw: Any = tomllib.load(stream)
        pack = LanguagePack.model_validate(raw)
        if pack.id in seen_ids:
            raise ValueError(f"duplicate language pack id: {pack.id}")
        normalized_tags = {normalize_language_tag(tag) for tag in pack.tags}
        duplicate_tags = normalized_tags & seen_tags
        if duplicate_tags:
            raise ValueError(f"duplicate language pack tags: {sorted(duplicate_tags)}")
        seen_ids.add(pack.id)
        seen_tags.update(normalized_tags)
        packs.append(pack)
    return tuple(packs)


def generation_guidance(language_tag: str) -> str:
    return " ".join(language_pack(language_tag).generation_guidance)


def language_pack_digest(language_tag: str) -> str:
    return language_pack(language_tag).digest()


def normalize_language_tag(value: str) -> str:
    return value.strip().replace("_", "-").casefold()


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


_SIGNATURE_TTL_SECONDS = 2.0
_rules_signature_cache: tuple[float, tuple[tuple[str, int, int], ...]] | None = None


def _rules_signature() -> tuple[tuple[str, int, int], ...]:
    """Detect edited rule files, stat-ing the directory at most every couple of seconds."""

    global _rules_signature_cache
    now = time.monotonic()
    if _rules_signature_cache is None or now - _rules_signature_cache[0] > _SIGNATURE_TTL_SECONDS:
        _rules_signature_cache = (now, _scan_rules_signature())
    return _rules_signature_cache[1]


def _scan_rules_signature() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in pack_files("rules")
    )


def _pattern_matches(
    pattern: LanguagePattern, lemma: str, normalized_pos: str, pack: LanguagePack
) -> bool:
    return (
        _pos_matches(pattern.pos, normalized_pos, pack)
        and re.fullmatch(pattern.lemma_regex, lemma) is not None
    )


def _pos_matches(allowed: tuple[str, ...], normalized_pos: str, pack: LanguagePack) -> bool:
    return not allowed or normalized_pos in pack.normalized_pos_set(allowed)


def _tag_matches(language_tag: str, configured_tag: str) -> bool:
    return language_tag == configured_tag or language_tag.startswith(f"{configured_tag}-")


def _merge_language_packs(language_tag: str, packs: list[LanguagePack]) -> LanguagePack:
    guidance: list[str] = []
    pos_aliases: dict[str, str] = {}
    overrides: dict[tuple[str, tuple[str, ...]], LearningOverride] = {}
    exceptions: dict[str, LanguagePattern] = {}
    exclusions: dict[str, LanguagePattern] = {}
    decompositions: dict[str, DecompositionPattern] = {}
    conventions: dict[str, LemmaConvention] = {}
    pronunciation_classes: dict[str, PronunciationClass] = {}
    contextual_pronunciations: dict[str, ContextualPronunciationRule] = {}
    surface_match = False
    pronunciation_ignores_whitespace = False
    for pack in packs:
        guidance.extend(pack.generation_guidance)
        pos_aliases.update(pack.pos_aliases)
        surface_match = pack.generated_surface_must_equal_lemma or surface_match
        pronunciation_ignores_whitespace = (
            pack.pronunciation_identity_ignores_whitespace or pronunciation_ignores_whitespace
        )
        overrides.update({(item.lemma, item.pos): item for item in pack.learning_overrides})
        exceptions.update({item.id: item for item in pack.learning_exceptions})
        exclusions.update({item.id: item for item in pack.non_learning_patterns})
        decompositions.update({item.id: item for item in pack.decomposition_patterns})
        conventions.update({item.id: item for item in pack.lemma_conventions})
        pronunciation_classes.update({item.id: item for item in pack.pronunciation_classes})
        contextual_pronunciations.update(
            {item.id: item for item in pack.contextual_pronunciation_rules}
        )
    most_specific = packs[-1]
    return LanguagePack(
        schema_version=1,
        id="-".join(pack.id for pack in packs),
        tags=(language_tag,),
        description=most_specific.description,
        generation_guidance=tuple(guidance),
        pos_aliases=pos_aliases,
        pronunciation_identity_ignores_whitespace=pronunciation_ignores_whitespace,
        generated_surface_must_equal_lemma=surface_match,
        learning_overrides=tuple(overrides.values()),
        learning_exceptions=tuple(exceptions.values()),
        non_learning_patterns=tuple(exclusions.values()),
        decomposition_patterns=tuple(decompositions.values()),
        lemma_conventions=tuple(conventions.values()),
        pronunciation_classes=tuple(pronunciation_classes.values()),
        contextual_pronunciation_rules=tuple(contextual_pronunciations.values()),
    )
