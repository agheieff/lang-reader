"""Sentence-local lexical generation and deterministic lesson-level identity assembly."""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from server.learning_units import generated_unit_errors, term_identity
from server.schemas import (
    CallbackLessonRun,
    GeneratedLessonBlock,
    GeneratedLessonDraft,
    GenerationLexicalConflictResult,
    GenerationLexicalResult,
    GenerationLexicalUnitResult,
    GenerationProseResult,
    LessonRun,
    LessonSentence,
    LessonTerm,
    LexicalConflictCandidate,
    LexicalConflictContext,
    LexicalConflictGroup,
    LexicalLessonBlock,
    LexicalLessonDraft,
    LexicalLessonSentence,
    ProseLessonSentence,
)


@dataclass(frozen=True)
class LexicalUnit:
    """One frozen title or body sentence in canonical lesson order."""

    unit_id: str
    lesson_index: int
    sentence_index: int
    is_title: bool
    block_index: int | None
    sentence: ProseLessonSentence
    learning_language: str


@dataclass
class _Candidate:
    candidate_id: str
    term: LessonTerm
    stored: bool
    contexts: list[LexicalConflictContext] = field(default_factory=list)


@dataclass
class _MergePlan:
    units: tuple[LexicalUnit, ...]
    results: dict[str, GenerationLexicalUnitResult]
    scoped_candidates: dict[tuple[str, str], str]
    candidates: dict[str, _Candidate]
    groups: list[LexicalConflictGroup]
    candidate_groups: dict[str, str]
    # Candidates the host merged without a model call because lemma, POS and pronunciation
    # agree and one set of glosses contains the other; maps a candidate to its anchor.
    auto_resolutions: dict[str, str] = field(default_factory=dict)


def enumerate_lexical_units(
    prose: GenerationProseResult,
    learning_language: str,
) -> tuple[LexicalUnit, ...]:
    units: list[LexicalUnit] = []
    for lesson_index, lesson in enumerate(prose.lessons):
        units.append(
            LexicalUnit(
                unit_id=f"lesson-{lesson_index + 1}-sentence-0",
                lesson_index=lesson_index,
                sentence_index=0,
                is_title=True,
                block_index=None,
                sentence=lesson.title_sentence,
                learning_language=learning_language,
            )
        )
        sentence_index = 1
        for block_index, block in enumerate(lesson.blocks):
            for sentence in block.sentences:
                units.append(
                    LexicalUnit(
                        unit_id=f"lesson-{lesson_index + 1}-sentence-{sentence_index}",
                        lesson_index=lesson_index,
                        sentence_index=sentence_index,
                        is_title=False,
                        block_index=block_index,
                        sentence=sentence,
                        learning_language=learning_language,
                    )
                )
                sentence_index += 1
    return tuple(units)


def validate_lexical_unit_result(
    unit: LexicalUnit,
    result: GenerationLexicalUnitResult,
    learning_language: str,
) -> None:
    if result.unit_id != unit.unit_id:
        raise ValueError(
            f"lexical unit ID changed: got {result.unit_id!r}, expected {unit.unit_id!r}"
        )
    if result.key != unit.sentence.key:
        raise ValueError(
            f"lexical sentence key changed: got {result.key!r}, expected {unit.sentence.key!r}"
        )
    reconstructed = "".join(run.text for run in result.runs)
    if reconstructed != unit.sentence.text:
        raise ValueError(f"lexical unit {unit.unit_id} did not reconstruct frozen source exactly")

    catalog = {term.key: term for term in result.terms}
    runs = [
        LessonRun(
            text=run.text,
            term=catalog.get(run.term_key) if run.term_key is not None else None,
            pronunciation=run.pronunciation,
        )
        for run in result.runs
    ]
    sentence = LessonSentence(key=result.key, runs=runs, translation="")
    dummy_key = f"__lexical-validation-{unit.lesson_index}-{unit.sentence_index}"
    if dummy_key == result.key:
        dummy_key += "-title"
    draft = GeneratedLessonDraft(
        title=".",
        title_sentence=LessonSentence(
            key=dummy_key,
            runs=[LessonRun(text=".")],
            translation="",
        ),
        topic=None,
        level="A1",
        difficulty=0.0,
        blocks=[
            GeneratedLessonBlock(
                key="__lexical-validation-block",
                sentences=[sentence],
            )
        ],
        target_term_keys=[],
    )
    errors = generated_unit_errors(draft, learning_language)
    if errors:
        raise ValueError(
            f"lexical unit {unit.unit_id} has invalid learning units: " + "; ".join(errors)
        )


def discover_lexical_conflicts(
    units: Sequence[LexicalUnit],
    results: Sequence[GenerationLexicalUnitResult],
    *,
    learning_language: str,
    stored_terms: Sequence[LessonTerm],
    offered_known_keys: Collection[str] = (),
) -> list[LexicalConflictGroup]:
    return _build_merge_plan(
        units,
        results,
        learning_language=learning_language,
        stored_terms=stored_terms,
        offered_known_keys=offered_known_keys,
    ).groups


def merge_lexical_unit_results(
    prose: GenerationProseResult,
    results: Sequence[GenerationLexicalUnitResult],
    *,
    task_id: int,
    learning_language: str,
    stored_terms: Sequence[LessonTerm],
    offered_known_keys: Collection[str] = (),
    conflict_result: GenerationLexicalConflictResult | None = None,
    conflict_results: Sequence[GenerationLexicalConflictResult] = (),
) -> GenerationLexicalResult:
    if conflict_result is not None and conflict_results:
        raise ValueError("pass either conflict_result or conflict_results, not both")
    units = enumerate_lexical_units(prose, learning_language)
    plan = _build_merge_plan(
        units,
        results,
        learning_language=learning_language,
        stored_terms=stored_terms,
        offered_known_keys=offered_known_keys,
    )
    supplied_results = tuple(conflict_results) if conflict_result is None else (conflict_result,)
    resolutions = _validated_resolutions(plan, supplied_results)
    # An anchor may itself have been merged by the model; follow it so no chain remains.
    resolutions.update(
        (candidate_id, resolutions.get(anchor_id, anchor_id))
        for candidate_id, anchor_id in plan.auto_resolutions.items()
    )
    canonical_terms = _allocate_canonical_terms(
        plan,
        resolutions,
        task_id=task_id,
        occupied_keys={term.key for term in stored_terms},
    )

    rewritten: dict[str, LexicalLessonSentence] = {}
    lesson_catalogs: list[dict[str, LessonTerm]] = [{} for _lesson in prose.lessons]
    for unit in plan.units:
        result = plan.results[unit.unit_id]
        runs: list[CallbackLessonRun] = []
        for run in result.runs:
            final_key: str | None = None
            if run.term_key is not None:
                candidate_id = plan.scoped_candidates[(unit.unit_id, run.term_key)]
                canonical_id = resolutions.get(candidate_id, candidate_id)
                term = canonical_terms[canonical_id]
                final_key = term.key
                lesson_catalogs[unit.lesson_index].setdefault(term.key, term)
            runs.append(
                CallbackLessonRun(
                    text=run.text,
                    term_key=final_key,
                    pronunciation=run.pronunciation,
                )
            )
        for local_term in result.terms:
            candidate_id = plan.scoped_candidates[(unit.unit_id, local_term.key)]
            canonical_id = resolutions.get(candidate_id, candidate_id)
            term = canonical_terms[canonical_id]
            lesson_catalogs[unit.lesson_index].setdefault(term.key, term)
        rewritten[unit.unit_id] = LexicalLessonSentence(
            key=unit.sentence.key,
            runs=runs,
        )

    lessons: list[LexicalLessonDraft] = []
    unit_by_position = {(unit.lesson_index, unit.sentence_index): unit for unit in plan.units}
    for lesson_index, prose_lesson in enumerate(prose.lessons):
        title_unit = unit_by_position[(lesson_index, 0)]
        body_index = 1
        blocks: list[LexicalLessonBlock] = []
        for prose_block in prose_lesson.blocks:
            sentences: list[LexicalLessonSentence] = []
            for _sentence in prose_block.sentences:
                unit = unit_by_position[(lesson_index, body_index)]
                sentences.append(rewritten[unit.unit_id])
                body_index += 1
            blocks.append(LexicalLessonBlock(key=prose_block.key, sentences=sentences))
        lessons.append(
            LexicalLessonDraft(
                terms=list(lesson_catalogs[lesson_index].values()),
                title_sentence=rewritten[title_unit.unit_id],
                blocks=blocks,
                target_term_keys=[],
            )
        )
    return GenerationLexicalResult(schema_version=1, lessons=lessons)


def validate_lexical_conflict_result(
    groups: Sequence[LexicalConflictGroup],
    result: GenerationLexicalConflictResult | None,
) -> dict[str, str]:
    candidates = {
        candidate.candidate_id: candidate for group in groups for candidate in group.candidates
    }
    candidate_groups = {
        candidate.candidate_id: group.group_id for group in groups for candidate in group.candidates
    }
    expected = {
        candidate_id for candidate_id, candidate in candidates.items() if not candidate.stored
    }
    if not expected:
        if result is not None and result.resolutions:
            raise ValueError("lexical conflict response was returned without conflicts")
        return {}
    if result is None:
        raise ValueError("lexical conflicts require an explicit reconciliation result")
    resolutions = {
        resolution.candidate_id: resolution.canonical_candidate_id
        for resolution in result.resolutions
    }
    if set(resolutions) != expected:
        missing = expected - resolutions.keys()
        extra = resolutions.keys() - expected
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise ValueError("lexical conflict resolutions are incomplete: " + "; ".join(details))
    for candidate_id, canonical_id in resolutions.items():
        candidate_group = candidate_groups[candidate_id]
        if candidate_groups.get(canonical_id) != candidate_group:
            raise ValueError(
                f"lexical conflict resolution crosses groups: {candidate_id} -> {canonical_id}"
            )
        canonical = candidates[canonical_id]
        if not canonical.stored and resolutions.get(canonical_id) != canonical_id:
            raise ValueError(
                "lexical conflict resolution must point directly to a self-canonical "
                f"candidate: {candidate_id} -> {canonical_id}"
            )
    return resolutions


def _build_merge_plan(
    units: Sequence[LexicalUnit],
    results: Sequence[GenerationLexicalUnitResult],
    *,
    learning_language: str,
    stored_terms: Sequence[LessonTerm],
    offered_known_keys: Collection[str],
) -> _MergePlan:
    ordered_units = tuple(units)
    results_by_id: dict[str, GenerationLexicalUnitResult] = {}
    units_by_id = {unit.unit_id: unit for unit in ordered_units}
    if len(units_by_id) != len(ordered_units):
        raise ValueError("lexical units contain duplicate IDs")
    for result in results:
        if result.unit_id in results_by_id:
            raise ValueError(f"duplicate lexical result for unit {result.unit_id}")
        unit = units_by_id.get(result.unit_id)
        if unit is None:
            raise ValueError(f"unexpected lexical result unit {result.unit_id}")
        validate_lexical_unit_result(unit, result, learning_language)
        results_by_id[result.unit_id] = result
    missing = [unit.unit_id for unit in ordered_units if unit.unit_id not in results_by_id]
    if missing:
        raise ValueError("missing lexical unit results: " + ", ".join(missing))

    stored_by_key: dict[str, tuple[str, LessonTerm]] = {}
    stored_by_identity: dict[tuple[str, str, str, str], tuple[str, LessonTerm]] = {}
    stored_by_lemma: dict[str, list[tuple[str, LessonTerm]]] = {}
    candidates: dict[str, _Candidate] = {}
    for index, term in enumerate(stored_terms, start=1):
        stored_candidate_id = f"stored-{index}"
        identity = term_identity(term, learning_language)
        canonical = stored_by_identity.setdefault(identity, (stored_candidate_id, term))
        stored_by_key.setdefault(term.key, canonical)
        if canonical[0] != stored_candidate_id:
            continue
        stored_by_lemma.setdefault(identity[0], []).append((stored_candidate_id, term))
        candidates[stored_candidate_id] = _Candidate(stored_candidate_id, term, True)

    offered = set(offered_known_keys)
    generated_by_identity: dict[tuple[str, str, str, str], str] = {}
    scoped_candidates: dict[tuple[str, str], str] = {}
    generated_counter = 0
    for unit in ordered_units:
        result = results_by_id[unit.unit_id]
        for term in result.terms:
            identity = term_identity(term, learning_language)
            same_key = stored_by_key.get(term.key)
            candidate_id: str
            if (
                term.key in offered
                and same_key is not None
                and term_identity(same_key[1], learning_language)[0] == identity[0]
            ):
                candidate_id = same_key[0]
            elif (stored := stored_by_identity.get(identity)) is not None:
                candidate_id = stored[0]
            elif (generated := generated_by_identity.get(identity)) is not None:
                candidate_id = generated
            else:
                generated_counter += 1
                candidate_id = f"generated-{generated_counter}"
                generated_by_identity[identity] = candidate_id
                candidates[candidate_id] = _Candidate(candidate_id, term, False)
            scoped_candidates[(unit.unit_id, term.key)] = candidate_id

        for run in result.runs:
            if run.term_key is None:
                continue
            candidate_id = scoped_candidates[(unit.unit_id, run.term_key)]
            context = LexicalConflictContext(
                sentence_key=unit.sentence.key,
                surface=run.text,
                sentence_text=unit.sentence.text,
            )
            if context not in candidates[candidate_id].contexts:
                candidates[candidate_id].contexts.append(context)

    generated_by_lemma: dict[str, list[_Candidate]] = {}
    for candidate in candidates.values():
        if candidate.stored or not candidate.contexts:
            continue
        identity = term_identity(candidate.term, learning_language)
        generated_by_lemma.setdefault(identity[0], []).append(candidate)

    groups: list[LexicalConflictGroup] = []
    candidate_groups: dict[str, str] = {}
    auto_resolutions: dict[str, str] = {}
    for lemma, lemma_candidates in generated_by_lemma.items():
        merged = _merge_equivalent_candidates(
            [*(candidates[cid] for cid, _ in stored_by_lemma.get(lemma, ())), *lemma_candidates],
            learning_language,
        )
        auto_resolutions.update(merged)
        generated_candidates = [c for c in lemma_candidates if c.candidate_id not in merged]
        if not generated_candidates:
            continue
        generated_identities = [
            term_identity(candidate.term, learning_language) for candidate in generated_candidates
        ]
        generated_pos = {identity[1] for identity in generated_identities}
        generated_pronunciations = {identity[3] for identity in generated_identities}
        stored_candidates = [
            candidates[candidate_id]
            for candidate_id, term in stored_by_lemma.get(lemma, ())
            if (
                (identity := term_identity(term, learning_language))[1] in generated_pos
                or identity[3] in generated_pronunciations
            )
        ]
        group_candidates = [*stored_candidates, *generated_candidates]
        if len(group_candidates) < 2:
            continue
        if len(generated_candidates) > 16:
            raise ValueError(
                "lexical identity conflict group exceeds 16 candidates for "
                f"{generated_candidates[0].term.lemma!r}"
            )
        group_candidates = [
            *stored_candidates[: 16 - len(generated_candidates)],
            *generated_candidates,
        ]
        group_id = f"conflict-{len(groups) + 1}"
        group = LexicalConflictGroup(
            group_id=group_id,
            candidates=[
                LexicalConflictCandidate(
                    candidate_id=candidate.candidate_id,
                    term=candidate.term,
                    stored=candidate.stored,
                    contexts=candidate.contexts,
                )
                for candidate in group_candidates
            ],
        )
        groups.append(group)
        for candidate in group_candidates:
            candidate_groups[candidate.candidate_id] = group_id

    return _MergePlan(
        units=ordered_units,
        results=results_by_id,
        scoped_candidates=scoped_candidates,
        candidates=candidates,
        groups=groups,
        candidate_groups=candidate_groups,
        auto_resolutions=auto_resolutions,
    )


def _senses(gloss: str) -> frozenset[str]:
    """The distinct senses of a dictionary gloss, ignoring case, articles and 'to'."""

    senses = set()
    for part in re.split(r"[;/]", gloss.lower()):
        words = re.sub(r"\([^)]*\)", " ", part)
        words = re.sub(r"[^\w\s]", " ", words)
        tokens = [token for token in words.split() if token not in {"to", "a", "an", "the"}]
        if tokens:
            senses.add(" ".join(tokens))
    return frozenset(senses)


def _merge_equivalent_candidates(
    candidates: Sequence[_Candidate], learning_language: str
) -> dict[str, str]:
    """Merge generated candidates into an anchor with the same POS, reading and sense.

    Conservative on purpose: anything a model might see as a different sense or reading
    (a different POS or pronunciation, or glosses that do not contain one another) stays
    unmerged and goes to the model-based reconciliation.
    """

    anchors: dict[tuple[str, str], list[_Candidate]] = {}
    merged: dict[str, str] = {}
    for candidate in candidates:
        _lemma, pos, _gloss, pronunciation = term_identity(candidate.term, learning_language)
        cell = anchors.setdefault((pos, pronunciation), [])
        senses = _senses(candidate.term.gloss)
        match = next(
            (
                anchor
                for anchor in cell
                if senses
                and (senses <= _senses(anchor.term.gloss) or _senses(anchor.term.gloss) <= senses)
            ),
            None,
        )
        if candidate.stored or match is None:
            cell.append(candidate)
        else:
            merged[candidate.candidate_id] = match.candidate_id
    return merged


def _validated_resolutions(
    plan: _MergePlan,
    results: Sequence[GenerationLexicalConflictResult],
) -> dict[str, str]:
    if len(results) <= 1:
        result = results[0] if results else None
        return validate_lexical_conflict_result(plan.groups, result)

    resolutions: dict[str, str] = {}
    for result in results:
        for resolution in result.resolutions:
            if resolution.candidate_id in resolutions:
                raise ValueError(
                    "lexical conflict results resolve a candidate more than once: "
                    f"{resolution.candidate_id}"
                )
            resolutions[resolution.candidate_id] = resolution.canonical_candidate_id
    expected = {
        candidate.candidate_id
        for group in plan.groups
        for candidate in group.candidates
        if not candidate.stored
    }
    if set(resolutions) != expected:
        missing = expected - resolutions.keys()
        extra = resolutions.keys() - expected
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise ValueError("lexical conflict resolutions are incomplete: " + "; ".join(details))
    candidate_groups = {
        candidate.candidate_id: group.group_id
        for group in plan.groups
        for candidate in group.candidates
    }
    candidates = {
        candidate.candidate_id: candidate for group in plan.groups for candidate in group.candidates
    }
    for candidate_id, canonical_id in resolutions.items():
        if candidate_groups.get(canonical_id) != candidate_groups[candidate_id]:
            raise ValueError(
                f"lexical conflict resolution crosses groups: {candidate_id} -> {canonical_id}"
            )
        canonical = candidates[canonical_id]
        if not canonical.stored and resolutions.get(canonical_id) != canonical_id:
            raise ValueError(
                "lexical conflict resolution must point directly to a self-canonical "
                f"candidate: {candidate_id} -> {canonical_id}"
            )
    return resolutions


def _allocate_canonical_terms(
    plan: _MergePlan,
    resolutions: dict[str, str],
    *,
    task_id: int,
    occupied_keys: set[str],
) -> dict[str, LessonTerm]:
    allocated: dict[str, LessonTerm] = {
        candidate_id: candidate.term
        for candidate_id, candidate in plan.candidates.items()
        if candidate.stored
    }
    counter = 0
    for unit in plan.units:
        result = plan.results[unit.unit_id]
        local_keys = [
            *(run.term_key for run in result.runs if run.term_key is not None),
            *(term.key for term in result.terms),
        ]
        for local_key in local_keys:
            candidate_id = plan.scoped_candidates[(unit.unit_id, local_key)]
            canonical_id = resolutions.get(candidate_id, candidate_id)
            if canonical_id in allocated:
                continue
            candidate = plan.candidates[canonical_id]
            if candidate.stored:
                allocated[canonical_id] = candidate.term
                continue
            while True:
                counter += 1
                key = f"generated-task-{task_id}-term-{counter}"
                if key not in occupied_keys:
                    break
            occupied_keys.add(key)
            allocated[canonical_id] = candidate.term.model_copy(update={"key": key})
    return allocated
