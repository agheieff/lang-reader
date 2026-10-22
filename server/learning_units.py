"""Conservative, language-aware boundaries for curriculum vocabulary.

Lesson runs remain the display and click boundary.  This module answers the narrower question of
whether a run's term is a reusable learning unit.  The distinction lets old phrase annotations stay
clickable without turning every productive combination into an SRS item.
"""

from __future__ import annotations

import threading
import unicodedata
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from server.language_packs import (
    LanguagePack,
    LearningComponent,
    LearningDecomposition,
    language_pack,
)
from server.lesson_content import document_fingerprint
from server.schemas import LessonDocument, LessonTerm

if TYPE_CHECKING:
    from server.schemas import GeneratedLessonDraft


class TermDefinition(Protocol):
    key: str
    lemma: str
    pos: str
    gloss: str
    pronunciation: str | None


TermIdentity = tuple[str, str, str, str]


@dataclass(frozen=True)
class LearningComponentCandidates:
    """Known dictionary entries that could receive evidence for one surface component."""

    component: LearningComponent
    canonical_keys: tuple[str, ...]


@dataclass(frozen=True)
class LearningUnitIndex:
    """Resolution from immutable display keys to canonical curriculum identities."""

    definitions: dict[str, LessonTerm]
    display_to_learning: dict[str, str | None]
    language_tag: str
    decompositions: dict[str, LearningDecomposition]
    candidates_by_display: dict[str, tuple[LearningComponentCandidates, ...]]

    def resolve(self, display_key: str) -> str | None:
        return self.display_to_learning.get(display_key)

    def learning_catalog(self) -> dict[str, LessonTerm]:
        return dict(self.definitions)

    def decomposition(self, display_key: str) -> LearningDecomposition | None:
        return self.decompositions.get(display_key)

    def component_candidates(self, display_key: str) -> tuple[LearningComponentCandidates, ...]:
        """Return all exact known entries for every configured component, in surface order."""

        return self.candidates_by_display.get(display_key, ())

    def select_evidence_targets(
        self, display_key: str, establishment: Mapping[str, float]
    ) -> tuple[str, ...]:
        """Choose one known sense per component, preferring replay-established definitions."""

        candidates = self.component_candidates(display_key)
        if not candidates or any(not item.canonical_keys for item in candidates):
            return ()
        return tuple(
            min(
                item.canonical_keys,
                key=lambda key: (-establishment.get(key, 0.0), key),
            )
            for item in candidates
        )


def learning_unit_issue(language_tag: str, term: TermDefinition) -> str | None:
    """Return why a term is unsuitable for vocabulary/SRS, or ``None`` when it is suitable.

    Language packs intentionally contain only high-confidence patterns. They do not perform fuzzy
    stemming or merge by lemma, which would damage real homographs and lexicalized compounds.
    """

    lemma = unicodedata.normalize("NFC", term.lemma)
    return language_pack(language_tag).learning_issue(lemma, term.pos)


def is_learning_unit(language_tag: str, term: TermDefinition) -> bool:
    return learning_unit_issue(language_tag, term) is None


def term_identity(term: TermDefinition, language_tag: str = "und") -> TermIdentity:
    """A conservative exact-definition identity; approximate frequency is deliberately excluded."""

    return _term_identity(term, language_pack(language_tag))


def _term_identity(term: TermDefinition, pack: LanguagePack) -> TermIdentity:
    return (
        _normalized_text(term.lemma),
        pack.normalize_pos(term.pos),
        _normalized_text(term.gloss),
        pack.normalize_pronunciation(term.pronunciation or ""),
    )


def learning_unit_indexes(
    documents: Iterable[LessonDocument],
) -> dict[tuple[str, str], LearningUnitIndex]:
    """Build one profile-local index per (learning, translation) language pair."""

    grouped: dict[tuple[str, str], list[LessonDocument]] = defaultdict(list)
    for document in documents:
        grouped[(document.learning_language, document.translation_language)].append(document)
    return {
        languages: build_learning_unit_index(profile_documents)
        for languages, profile_documents in grouped.items()
    }


_MAX_CACHED_INDEXES = 16
_indexes: OrderedDict[tuple[object, ...], tuple[LanguagePack, LearningUnitIndex]] = OrderedDict()
_indexes_lock = threading.Lock()


def build_learning_unit_index(documents: Iterable[LessonDocument]) -> LearningUnitIndex:
    """Build a profile-local index in caller-supplied (normally import) order.

    Indexes over cached documents are reused while the language pack is unchanged; callers must
    treat the returned index as immutable.
    """

    document_list = list(documents)
    fingerprints = [document_fingerprint(document) for document in document_list]
    if not document_list or any(value is None for value in fingerprints):
        return _build_learning_unit_index(document_list)
    languages = {document.learning_language for document in document_list}
    pack = language_pack(next(iter(languages))) if len(languages) == 1 else None
    key = (tuple(sorted(languages)), tuple(fingerprints))
    with _indexes_lock:
        cached = _indexes.get(key)
        if cached is not None and cached[0] is pack:
            _indexes.move_to_end(key)
            return cached[1]
    index = _build_learning_unit_index(document_list)
    if pack is not None:
        with _indexes_lock:
            _indexes[key] = (pack, index)
            while len(_indexes) > _MAX_CACHED_INDEXES:
                _indexes.popitem(last=False)
    return index


def _build_learning_unit_index(document_list: list[LessonDocument]) -> LearningUnitIndex:
    languages = {document.learning_language for document in document_list}
    if len(languages) > 1:
        raise ValueError("a learning-unit index must contain one learning language")
    language_tag = next(iter(languages), "und")
    pack = language_pack(language_tag)
    definitions: dict[str, LessonTerm] = {}
    display_to_learning: dict[str, str | None] = {}
    canonical_by_identity: dict[TermIdentity, str] = {}
    identity_by_key: dict[str, TermIdentity] = {}
    display_terms: dict[str, LessonTerm] = {}
    for document in document_list:
        for term in document.term_catalog().values():
            identity = _term_identity(term, pack)
            previous = identity_by_key.get(term.key)
            if previous is not None and previous != identity:
                raise ValueError(f"term key has conflicting definitions: {term.key}")
            identity_by_key[term.key] = identity
            display_terms.setdefault(term.key, term)
            if pack.learning_issue(unicodedata.normalize("NFC", term.lemma), term.pos) is not None:
                display_to_learning[term.key] = None
                continue
            canonical = canonical_by_identity.setdefault(identity, term.key)
            display_to_learning[term.key] = canonical
            definitions.setdefault(canonical, term)

    decompositions = {
        key: decomposition
        for key, term in display_terms.items()
        if (decomposition := pack.decompose(term.lemma, term.pos)) is not None
    }
    definitions_by_lemma: dict[str, list[tuple[str, LessonTerm]]] = {}
    for canonical, definition in definitions.items():
        definitions_by_lemma.setdefault(_normalized_text(definition.lemma), []).append(
            (canonical, definition)
        )
    candidates_by_display = {
        key: tuple(
            LearningComponentCandidates(
                component=component,
                canonical_keys=tuple(
                    canonical
                    for canonical, definition in definitions_by_lemma.get(
                        _normalized_text(component.surface), ()
                    )
                    if _component_pos_matches(component, definition, pack)
                ),
            )
            for component in decomposition.components
        )
        for key, decomposition in decompositions.items()
    }
    return LearningUnitIndex(
        definitions,
        display_to_learning,
        language_tag,
        decompositions,
        candidates_by_display,
    )


def generated_unit_errors(draft: GeneratedLessonDraft, learning_language: str) -> list[str]:
    """Return unsafe generated-unit errors the host cannot resolve deterministically.

    A configured decomposition is recoverable: the run remains the display/click boundary while
    curriculum evidence is projected onto its atomic components.  Generation guidance still asks
    agents for atomic runs, but a known productive span no longer discards an otherwise valid
    lesson.  Unknown non-learning patterns and malformed surface/lemma pairs remain errors.
    """

    errors: list[str] = []
    sentences = [
        draft.title_sentence,
        *(sentence for block in draft.blocks for sentence in block.sentences),
    ]
    seen: set[tuple[str, str]] = set()
    pack = language_pack(learning_language)
    for sentence in sentences:
        for run in sentence.runs:
            if run.term is None:
                continue
            issue = learning_unit_issue(learning_language, run.term)
            marker = (run.term.key, issue or "")
            decomposition = (
                pack.decompose(run.term.lemma, run.term.pos) if issue is not None else None
            )
            if issue is not None and decomposition is None and marker not in seen:
                seen.add(marker)
                errors.append(
                    f"{run.text!r} ({run.term.key}) is a {issue}; split it into adjacent reusable "
                    "learning-unit runs"
                )
            if pack.generated_surface_must_equal_lemma and unicodedata.normalize(
                "NFC", run.text
            ) != unicodedata.normalize("NFC", run.term.lemma):
                error = "generated surface text must equal its canonical lemma in this language"
                marker = (run.term.key, error)
                if marker not in seen:
                    seen.add(marker)
                    errors.append(f"{run.text!r} ({run.term.key}) {error}")
            for error in pack.lemma_errors(run.term.lemma, run.term.pos):
                marker = (run.term.key, error)
                if marker in seen:
                    continue
                seen.add(marker)
                errors.append(f"{run.text!r} ({run.term.key}) has a non-canonical lemma: {error}")
    return errors


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _component_pos_matches(
    component: LearningComponent, term: TermDefinition, pack: LanguagePack
) -> bool:
    return not component.pos or pack.normalize_pos(term.pos) in {
        pack.normalize_pos(allowed_pos) for allowed_pos in component.pos
    }
