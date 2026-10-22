"""Conservative normalization of compact callback runs before lesson expansion."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from server.language_packs import LanguagePack, LearningComponent, language_pack, normalize_text
from server.learning_units import term_identity
from server.schemas import (
    CallbackLessonBlock,
    CallbackLessonDraft,
    CallbackLessonRun,
    CallbackLessonSentence,
    LessonTerm,
)

_MAX_SENTENCE_RUNS = 100


def normalize_callback_lesson(
    lesson: CallbackLessonDraft,
    learning_language: str,
    fallback_terms: Iterable[LessonTerm] = (),
) -> CallbackLessonDraft:
    """Apply deterministic run splitting and configured contextual surface readings."""

    pack = language_pack(learning_language)
    catalog = lesson.term_catalog()
    fallback = tuple(fallback_terms)
    resolved = _resolve_missing_terms(
        lesson,
        catalog=catalog,
        fallback=fallback,
        pack=pack,
        learning_language=learning_language,
    )
    catalog.update(resolved)
    protected = (
        {probe.term_key for probe in lesson.calibration.probes}
        if lesson.calibration is not None
        else set()
    )
    added = dict(resolved)

    def normalize(sentence: CallbackLessonSentence) -> CallbackLessonSentence:
        normalized, additions = _normalize_sentence(
            sentence,
            catalog={**catalog, **added},
            fallback=fallback,
            protected=protected,
            pack=pack,
            learning_language=learning_language,
        )
        added.update({term.key: term for term in additions})
        return normalized

    title_sentence = normalize(lesson.title_sentence)
    blocks = [
        CallbackLessonBlock(
            key=block.key,
            sentences=[normalize(sentence) for sentence in block.sentences],
        )
        for block in lesson.blocks
    ]
    if not added and title_sentence == lesson.title_sentence and blocks == lesson.blocks:
        return lesson
    return lesson.model_copy(
        update={
            "terms": [*lesson.terms, *added.values()],
            "title_sentence": title_sentence,
            "blocks": blocks,
        }
    )


def callback_reference_errors(lesson: CallbackLessonDraft) -> list[str]:
    """Locate every unresolved compact run after deterministic normalization."""

    catalog = lesson.term_catalog()
    return [
        f"unknown term key {run.term_key!r} at sentence {sentence.key!r}, "
        f"run {run_index}, surface {run.text!r}"
        for sentence in [
            lesson.title_sentence,
            *(sentence for block in lesson.blocks for sentence in block.sentences),
        ]
        for run_index, run in enumerate(sentence.runs)
        if run.term_key is not None and run.term_key not in catalog
    ]


def _normalize_sentence(
    sentence: CallbackLessonSentence,
    *,
    catalog: dict[str, LessonTerm],
    fallback: tuple[LessonTerm, ...],
    protected: set[str],
    pack: LanguagePack,
    learning_language: str,
) -> tuple[CallbackLessonSentence, tuple[LessonTerm, ...]]:
    runs: list[CallbackLessonRun] = []
    additions: dict[str, LessonTerm] = {}
    boundaries = [0]
    for run in sentence.runs:
        replacement = _replacement_runs(
            run,
            catalog={**catalog, **additions},
            fallback=fallback,
            protected=protected,
            pack=pack,
            learning_language=learning_language,
        )
        if replacement is None:
            runs.append(run)
        else:
            replacement_runs, replacement_terms = replacement
            runs.extend(replacement_runs)
            additions.update({term.key: term for term in replacement_terms})
        boundaries.append(len(runs))

    replacements_applied = runs != sentence.runs and len(runs) <= _MAX_SENTENCE_RUNS
    if len(runs) > _MAX_SENTENCE_RUNS:
        runs = list(sentence.runs)
        additions.clear()
    runs = _apply_contextual_pronunciations(
        runs,
        catalog={**catalog, **additions},
        pack=pack,
    )
    if runs == sentence.runs:
        return sentence, ()
    grammar = (
        [
            occurrence.model_copy(
                update={
                    "run_start": boundaries[occurrence.run_start],
                    "run_end": boundaries[occurrence.run_end],
                }
            )
            for occurrence in sentence.grammar
        ]
        if replacements_applied
        else sentence.grammar
    )
    normalized = sentence.model_copy(update={"runs": runs, "grammar": grammar})
    if "".join(run.text for run in normalized.runs) != "".join(run.text for run in sentence.runs):
        raise AssertionError("callback run normalization changed sentence text")
    return normalized, tuple(additions.values())


def _apply_contextual_pronunciations(
    runs: list[CallbackLessonRun],
    *,
    catalog: dict[str, LessonTerm],
    pack: LanguagePack,
) -> list[CallbackLessonRun]:
    projected = list(runs)
    for index in range(len(projected) - 1, -1, -1):
        run = projected[index]
        term = catalog.get(run.term_key) if run.term_key is not None else None
        following = _following_lexical_pronunciation(projected, index, catalog)
        if term is None or following is None:
            continue
        pronunciation = pack.contextual_pronunciation(
            lemma=term.lemma,
            pos=term.pos,
            canonical_pronunciation=term.pronunciation,
            following_pronunciation=following,
        )
        if pronunciation is None:
            continue
        expected = (
            None
            if term.pronunciation is not None
            and pack.normalize_pronunciation(pronunciation)
            == pack.normalize_pronunciation(term.pronunciation)
            else pronunciation
        )
        if expected is None:
            if run.pronunciation is not None:
                projected[index] = run.model_copy(update={"pronunciation": None})
        elif run.pronunciation is None or pack.normalize_pronunciation(
            run.pronunciation
        ) != pack.normalize_pronunciation(expected):
            projected[index] = run.model_copy(update={"pronunciation": expected})
    return projected


def _following_lexical_pronunciation(
    runs: list[CallbackLessonRun],
    index: int,
    catalog: dict[str, LessonTerm],
) -> str | None:
    for run in runs[index + 1 :]:
        if run.term_key is not None:
            term = catalog.get(run.term_key)
            return run.pronunciation or (term.pronunciation if term is not None else None)
        if not run.text.isspace():
            return None
    return None


def _replacement_runs(
    run: CallbackLessonRun,
    *,
    catalog: dict[str, LessonTerm],
    fallback: tuple[LessonTerm, ...],
    protected: set[str],
    pack: LanguagePack,
    learning_language: str,
) -> tuple[tuple[CallbackLessonRun, ...], tuple[LessonTerm, ...]] | None:
    if run.term_key is None or run.term_key in protected:
        return None
    attached = catalog.get(run.term_key)
    if attached is not None:
        decomposition = _decomposition_replacement(
            run,
            attached=attached,
            catalog=catalog,
            fallback=fallback,
            protected=protected,
            pack=pack,
            learning_language=learning_language,
        )
        if decomposition is not None:
            return decomposition
        if _same_text(run.text, attached.lemma):
            return None
    return _surface_rebinding(
        run,
        catalog=catalog,
        fallback=fallback,
        protected=protected,
        pack=pack,
        learning_language=learning_language,
    )


def _decomposition_replacement(
    run: CallbackLessonRun,
    *,
    attached: LessonTerm,
    catalog: dict[str, LessonTerm],
    fallback: tuple[LessonTerm, ...],
    protected: set[str],
    pack: LanguagePack,
    learning_language: str,
) -> tuple[tuple[CallbackLessonRun, ...], tuple[LessonTerm, ...]] | None:
    exact_surface = _same_text(run.text, attached.lemma)
    decomposition = pack.decompose(attached.lemma if exact_surface else run.text, attached.pos)
    if decomposition is None:
        return None
    if run.pronunciation is not None and not (
        exact_surface and decomposition.recompute_contextual_pronunciation
    ):
        return None
    components = decomposition.components
    if "".join(component.surface for component in components) != run.text:
        return None
    attached_components = sum(
        _component_matches(component, attached, pack) for component in components
    )
    if not exact_surface and attached_components != 1:
        return None

    selected: list[LessonTerm] = []
    for component in components:
        local = [term for term in catalog.values() if _component_matches(component, term, pack)]
        candidates = local or [
            term for term in fallback if _component_matches(component, term, pack)
        ]
        term = _one_identity(candidates, attached.key, learning_language)
        if term is None or term.key in protected:
            return None
        collision = catalog.get(term.key)
        if collision is not None and term_identity(collision, learning_language) != term_identity(
            term, learning_language
        ):
            return None
        prior = next((item for item in selected if item.key == term.key), None)
        if prior is not None and term_identity(prior, learning_language) != term_identity(
            term, learning_language
        ):
            return None
        selected.append(term)

    return (
        tuple(
            CallbackLessonRun(text=component.surface, term_key=term.key)
            for component, term in zip(components, selected, strict=True)
        ),
        tuple(term for term in selected if term.key not in catalog),
    )


def _surface_rebinding(
    run: CallbackLessonRun,
    *,
    catalog: dict[str, LessonTerm],
    fallback: tuple[LessonTerm, ...],
    protected: set[str],
    pack: LanguagePack,
    learning_language: str,
) -> tuple[tuple[CallbackLessonRun, ...], tuple[LessonTerm, ...]] | None:
    local = [term for term in catalog.values() if _same_text(run.text, term.lemma)]
    candidates = local or [term for term in fallback if _same_text(run.text, term.lemma)]
    term = _one_identity(candidates, run.term_key or "", learning_language)
    if term is None or term.key in protected:
        return None
    collision = catalog.get(term.key)
    if collision is not None and term_identity(collision, learning_language) != term_identity(
        term, learning_language
    ):
        return None
    if run.pronunciation is not None and (
        term.pronunciation is None
        or pack.normalize_pronunciation(run.pronunciation)
        != pack.normalize_pronunciation(term.pronunciation)
    ):
        return None
    return (
        (
            CallbackLessonRun(
                text=run.text,
                term_key=term.key,
                pronunciation=run.pronunciation,
            ),
        ),
        () if term.key in catalog else (term,),
    )


def _resolve_missing_terms(
    lesson: CallbackLessonDraft,
    *,
    catalog: dict[str, LessonTerm],
    fallback: tuple[LessonTerm, ...],
    pack: LanguagePack,
    learning_language: str,
) -> dict[str, LessonTerm]:
    runs = [
        run
        for sentence in [
            lesson.title_sentence,
            *(sentence for block in lesson.blocks for sentence in block.sentences),
        ]
        for run in sentence.runs
    ]
    missing = {
        run.term_key for run in runs if run.term_key is not None and run.term_key not in catalog
    }
    resolved: dict[str, LessonTerm] = {}
    for key in sorted(missing):
        term = _one_identity(
            [candidate for candidate in fallback if candidate.key == key],
            key,
            learning_language,
        )
        references = [run for run in runs if run.term_key == key]
        if term is None or (
            pack.generated_surface_must_equal_lemma
            and any(not _same_text(run.text, term.lemma) for run in references)
        ):
            continue
        resolved[key] = term
    return resolved


def _one_identity(
    candidates: list[LessonTerm], preferred_key: str, learning_language: str
) -> LessonTerm | None:
    identities = {term_identity(term, learning_language) for term in candidates}
    if len(identities) != 1:
        return None
    return min(candidates, key=lambda term: (term.key != preferred_key, term.key))


def _component_matches(component: LearningComponent, term: LessonTerm, pack: LanguagePack) -> bool:
    return _same_text(component.surface, term.lemma) and (
        not component.pos
        or pack.normalize_pos(term.pos)
        in {pack.normalize_pos(allowed) for allowed in component.pos}
    )


def _same_text(left: str, right: str) -> bool:
    return normalize_text(unicodedata.normalize("NFC", left)) == normalize_text(
        unicodedata.normalize("NFC", right)
    )
