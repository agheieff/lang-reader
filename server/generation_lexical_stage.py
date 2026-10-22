"""Sentence-level lexical generation, repair, and identity reconciliation."""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import ValidationError

from server.generation_callback import (
    CallbackInvocation,
    CallbackResult,
    GenerationCallback,
    PipelineState,
    completed_stage_result,
    make_stage_invocation,
    prose_sha256,
    run_timed_callback,
    stage_request_common,
    write_private,
)
from server.lexical_generation import (
    LexicalUnit,
    discover_lexical_conflicts,
    enumerate_lexical_units,
    merge_lexical_unit_results,
    validate_lexical_conflict_result,
    validate_lexical_unit_result,
)
from server.schemas import (
    MAX_LEXICAL_BATCH_UNITS,
    GenerationCallbackRequest,
    GenerationLexicalBatchRequest,
    GenerationLexicalConflictRequest,
    GenerationLexicalConflictResult,
    GenerationLexicalResult,
    GenerationLexicalUnitRequest,
    GenerationLexicalUnitResult,
    GenerationProseResult,
    LessonTerm,
    LexicalBatchUnit,
    LexicalConflictGroup,
    ProseLessonSentence,
)

TERM_IDENTITY_PREFLIGHT_INSTRUCTIONS = (
    "Required final term-identity preflight before returning JSON: verify that every term_key in "
    "every title and body run resolves to exactly one entry in that lesson's terms catalog, every "
    "catalog key is defined at most once. Extra catalog entries are allowed only when they define "
    "atomic components of a decomposable display span. If a key is reused across lessons in one "
    "response, its lemma, pos, gloss, pronunciation, and frequency_rank must be identical. "
    "Verify that every visible run surface is genuinely an occurrence of its referenced lemma; "
    "when the language guidance requires generated surface text to equal the lemma, compare them "
    "exactly. Never attach an unrelated existing key merely to avoid defining the visible word. "
    "Verify separately that every non-name catalog term has an integer frequency_rank of at least "
    "1; only proper names may use null. If the same written form is used with a different POS or "
    "meaning, give each sense a different key; for example, a noun meaning 'forecast' and a verb "
    "meaning 'to forecast' must not share a key."
)

MAX_UNMATCHED_PRIORITY_TERMS_PER_UNIT = 8
MAX_KNOWN_TERMS_PER_UNIT = 24
MAX_REPAIR_DIAGNOSTIC_CHARACTERS = 2_000


@dataclass(frozen=True)
class LexicalExecutionPolicy:
    """Operational bounds; these do not change lesson semantics."""

    parallel_callbacks: int = 2
    local_repair_limit: int = 1
    conflict_chunk_size: int = 64
    # Sentences per lexical callback. Each callback carries a large fixed instruction and agent
    # overhead, so per-sentence calls spent most tokens on repetition; 1 restores that mode.
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.parallel_callbacks < 1:
            raise ValueError("parallel_callbacks must be positive")
        if not 1 <= self.batch_size <= MAX_LEXICAL_BATCH_UNITS:
            raise ValueError("batch_size must be between 1 and the batch schema limit")
        if self.local_repair_limit < 0:
            raise ValueError("local_repair_limit must not be negative")
        if self.conflict_chunk_size < 1:
            raise ValueError("conflict_chunk_size must be positive")


DEFAULT_LEXICAL_EXECUTION_POLICY = LexicalExecutionPolicy()


@dataclass
class LexicalStageContext:
    """Host state needed to execute and persist one lexical stage."""

    task_id: int
    prose: GenerationProseResult
    request: GenerationCallbackRequest
    state: PipelineState
    manifest: CallbackInvocation
    log_path: Path
    logs: list[str]
    stored_terms: Sequence[LessonTerm]


ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class _RepairJob(Generic[ItemT, ResultT]):
    item: ItemT
    repair: ResultT | None = None
    attempt: int = 1
    error: str | None = None


@dataclass(frozen=True)
class _PendingJob(Generic[ItemT, ResultT]):
    item: ItemT
    invocation: CallbackInvocation
    attempt: int
    retained_repair: ResultT | None


def _run_repair_jobs(
    callback: GenerationCallback,
    executor: ThreadPoolExecutor,
    jobs: Sequence[_RepairJob[ItemT, ResultT]],
    *,
    policy: LexicalExecutionPolicy,
    make_invocation: Callable[
        [ItemT, int, ResultT | None, str | None],
        CallbackInvocation,
    ],
    parse_result: Callable[[ItemT, str], ResultT],
    validate_result: Callable[[ItemT, ResultT], None],
    accept_result: Callable[[ItemT, ResultT], None],
    retain_failure: Callable[[ItemT, ResultT | None], None],
    describe: Callable[[ItemT], str],
    log_path: Path,
    logs: list[str],
) -> list[str]:
    """Run bounded jobs with the same one-result repair lifecycle."""

    queue = deque(jobs)
    pending: dict[Future[CallbackResult], _PendingJob[ItemT, ResultT]] = {}

    def submit(
        item: ItemT,
        *,
        attempt: int,
        repair: ResultT | None,
        retained_repair: ResultT | None,
        error: str | None,
    ) -> None:
        invocation = make_invocation(item, attempt, repair, error)
        queued_at = time.perf_counter()
        future = executor.submit(run_timed_callback, callback, invocation, queued_at)
        pending[future] = _PendingJob(
            item=item,
            invocation=invocation,
            attempt=attempt,
            retained_repair=retained_repair,
        )

    failures: list[str] = []
    while queue or pending:
        while queue and len(pending) < policy.parallel_callbacks:
            job = queue.popleft()
            submit(
                job.item,
                attempt=job.attempt,
                repair=job.repair,
                retained_repair=job.repair,
                error=job.error,
            )
        done, _not_done = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in done:
            current = pending.pop(future)
            stem = current.invocation.artifact_stem or "lexical"
            stage_result, callback_error = completed_stage_result(
                stem,
                future,
                current.invocation,
                log_path,
                logs,
            )
            parsed: ResultT | None = None
            error = callback_error
            if error is None and stage_result is not None:
                try:
                    parsed = parse_result(current.item, stage_result.payload)
                    validate_result(current.item, parsed)
                except (ValidationError, ValueError) as validation_error:
                    error = f"invalid response: {validation_error}"
            if error is None and parsed is not None:
                accept_result(current.item, parsed)
                continue

            retained = parsed if parsed is not None else current.retained_repair
            if current.attempt <= policy.local_repair_limit:
                submit(
                    current.item,
                    attempt=current.attempt + 1,
                    repair=retained,
                    retained_repair=retained,
                    error=error,
                )
                continue
            retain_failure(current.item, retained)
            failures.append(f"{describe(current.item)}: {error or 'unknown callback failure'}")
    return failures


def run_sentence_lexical_stage(
    callback: GenerationCallback,
    executor: ThreadPoolExecutor,
    context: LexicalStageContext,
    *,
    policy: LexicalExecutionPolicy = DEFAULT_LEXICAL_EXECUTION_POLICY,
) -> GenerationLexicalResult:
    """Tokenize all frozen sentences, reusing durable partial results when valid."""

    language = context.request.brief.profile.learning_language
    units = enumerate_lexical_units(context.prose, language)
    units_by_id = {unit.unit_id: unit for unit in units}
    if len(units_by_id) != len(units):
        raise ValueError("lexical unit enumeration produced duplicate IDs")

    results = {
        unit_id: result
        for unit_id, result in context.state.lexical_units.items()
        if unit_id in units_by_id
    }
    repairs = {
        unit_id: result
        for unit_id, result in context.state.lexical_unit_repairs.items()
        if unit_id in units_by_id and unit_id not in results
    }
    for unit_id, result in list(results.items()):
        try:
            validate_lexical_unit_result(units_by_id[unit_id], result, language)
        except ValueError:
            results.pop(unit_id)

    artifact_dir = context.manifest.response_path.parent
    partial_path = artifact_dir / "lexical-partial.json"
    pending_jobs: list[_RepairJob[LexicalUnit, GenerationLexicalUnitResult]] = []
    for unit in units:
        reused = results.get(unit.unit_id)
        if reused is None:
            pending_jobs.append(_RepairJob(unit, repairs.get(unit.unit_id)))
            continue
        invocation = _make_unit_invocation(
            context,
            unit,
            unit_attempt=1,
            repair_result=None,
            repair_error=None,
        )
        write_private(invocation.response_path, reused.model_dump_json())
        message = "reused validated sentence output from the preceding task attempt"
        write_private(
            artifact_dir / f"{invocation.artifact_stem}-callback.log",
            f"{message}\n",
        )
        context.logs.append(f"[{invocation.artifact_stem}]\n{message}")

    def make_invocation(
        unit: LexicalUnit,
        attempt: int,
        repair: GenerationLexicalUnitResult | None,
        error: str | None,
    ) -> CallbackInvocation:
        return _make_unit_invocation(
            context,
            unit,
            unit_attempt=attempt,
            repair_result=repair,
            repair_error=error,
        )

    def parse_result(
        _unit: LexicalUnit,
        payload: str,
    ) -> GenerationLexicalUnitResult:
        return GenerationLexicalUnitResult.model_validate_json(payload)

    def validate_result(
        unit: LexicalUnit,
        result: GenerationLexicalUnitResult,
    ) -> None:
        validate_lexical_unit_result(unit, result, language)

    def accept_result(unit: LexicalUnit, result: GenerationLexicalUnitResult) -> None:
        results[unit.unit_id] = result
        repairs.pop(unit.unit_id, None)
        _write_lexical_partial(
            partial_path,
            context.prose,
            context.request,
            units,
            results,
            repairs,
        )

    def retain_failure(
        unit: LexicalUnit,
        repair: GenerationLexicalUnitResult | None,
    ) -> None:
        if repair is not None:
            repairs[unit.unit_id] = repair
        _write_lexical_partial(
            partial_path,
            context.prose,
            context.request,
            units,
            results,
            repairs,
        )

    if policy.batch_size > 1:
        fresh_units = [job.item for job in pending_jobs if job.repair is None]
        carried = [job for job in pending_jobs if job.repair is not None]
        pending_jobs = carried + _run_lexical_batches(
            callback,
            executor,
            context,
            fresh_units,
            policy=policy,
            language=language,
            accept_result=accept_result,
        )

    failures = _run_repair_jobs(
        callback,
        executor,
        pending_jobs,
        policy=policy,
        make_invocation=make_invocation,
        parse_result=parse_result,
        validate_result=validate_result,
        accept_result=accept_result,
        retain_failure=retain_failure,
        describe=lambda unit: unit.unit_id,
        log_path=context.log_path,
        logs=context.logs,
    )
    if failures:
        raise ValueError(
            f"sentence jobs failed after {policy.local_repair_limit} local repair(s): "
            + "; ".join(failures)
        )
    if set(results) != set(units_by_id):
        missing = sorted(set(units_by_id) - set(results))
        raise ValueError("lexical sentence results are incomplete: " + ", ".join(missing))

    ordered_results = [results[unit.unit_id] for unit in units]
    offered_keys = {
        term.key for unit in units for term in _lexical_unit_known_terms(context.request, unit)
    }
    groups = discover_lexical_conflicts(
        units,
        ordered_results,
        learning_language=language,
        stored_terms=context.stored_terms,
        offered_known_keys=offered_keys,
    )
    conflict_results = (
        run_lexical_conflict_chunks(
            callback,
            executor,
            context.request,
            groups,
            job_dir=context.manifest.job_dir,
            artifact_dir=artifact_dir,
            state=context.state,
            log_path=context.log_path,
            logs=context.logs,
            policy=policy,
        )
        if groups
        else ()
    )
    merged = merge_lexical_unit_results(
        context.prose,
        ordered_results,
        task_id=context.task_id,
        learning_language=language,
        stored_terms=context.stored_terms,
        offered_known_keys=offered_keys,
        conflict_results=conflict_results,
    )

    write_private(context.manifest.response_path, merged.model_dump_json())
    write_private(
        artifact_dir / "lexical-callback.log",
        f"assembled {len(units)} validated sentence callbacks\n",
    )
    context.state.lexical_units = dict(results)
    context.state.lexical_unit_repairs = {}
    _write_lexical_partial(
        partial_path,
        context.prose,
        context.request,
        units,
        results,
        {},
    )
    return merged


def _run_lexical_batches(
    callback: GenerationCallback,
    executor: ThreadPoolExecutor,
    context: LexicalStageContext,
    units: Sequence[LexicalUnit],
    *,
    policy: LexicalExecutionPolicy,
    language: str,
    accept_result: Callable[[LexicalUnit, GenerationLexicalUnitResult], None],
) -> list[_RepairJob[LexicalUnit, GenerationLexicalUnitResult]]:
    """Tokenize sentences in batches; return per-sentence jobs for anything not accepted.

    Each returned unit is validated exactly like a single-sentence response. An invalid unit goes
    straight to its one sentence-local repair; a missing unit (or a failed batch) gets a fresh
    single-sentence attempt, so batching never lowers the per-sentence repair budget.
    """

    queue = deque(enumerate(_lexical_batches(units, policy.batch_size)))
    pending: dict[Future[CallbackResult], tuple[list[LexicalUnit], CallbackInvocation]] = {}
    leftovers: list[_RepairJob[LexicalUnit, GenerationLexicalUnitResult]] = []
    while queue or pending:
        while queue and len(pending) < policy.parallel_callbacks:
            index, batch = queue.popleft()
            invocation = make_stage_invocation(
                _lexical_batch_request(context, batch),
                base_request=context.request,
                job_dir=context.manifest.job_dir,
                artifact_dir=context.manifest.response_path.parent,
                pipeline_state=context.state,
                artifact_stem=f"lexical-batch-{index:02d}",
            )
            future = executor.submit(run_timed_callback, callback, invocation, time.perf_counter())
            pending[future] = (batch, invocation)
        done, _not_done = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in done:
            batch, invocation = pending.pop(future)
            stage_result, error = completed_stage_result(
                invocation.artifact_stem or "lexical-batch",
                future,
                invocation,
                context.log_path,
                context.logs,
            )
            returned: dict[str, object] = {}
            if error is None and stage_result is not None:
                try:
                    returned = _batch_units_by_id(stage_result.payload)
                except ValueError:
                    returned = {}
            for unit in batch:
                raw = returned.get(unit.unit_id)
                try:
                    parsed = GenerationLexicalUnitResult.model_validate(raw)
                except ValidationError:
                    leftovers.append(_RepairJob(unit))
                    continue
                try:
                    validate_lexical_unit_result(unit, parsed, language)
                except ValueError as invalid:
                    leftovers.append(
                        _RepairJob(unit, parsed, attempt=2, error=f"invalid response: {invalid}")
                    )
                    continue
                accept_result(unit, parsed)
    return leftovers


def _lexical_batches(units: Sequence[LexicalUnit], size: int) -> list[list[LexicalUnit]]:
    by_lesson: dict[int, list[LexicalUnit]] = {}
    for unit in units:
        by_lesson.setdefault(unit.lesson_index, []).append(unit)
    return [
        lesson_units[index : index + size]
        for lesson_units in by_lesson.values()
        for index in range(0, len(lesson_units), size)
    ]


def _batch_units_by_id(payload: str) -> dict[str, object]:
    data = json.loads(payload)
    units = data.get("units") if isinstance(data, dict) else None
    if not isinstance(units, list):
        raise ValueError("lexical batch response has no units list")
    result: dict[str, object] = {}
    for unit in units:
        if isinstance(unit, dict) and isinstance(unit.get("unit_id"), str):
            result.setdefault(unit["unit_id"].strip(), unit)
    return result


def _lexical_batch_request(
    context: LexicalStageContext,
    batch: Sequence[LexicalUnit],
) -> GenerationLexicalBatchRequest:
    known_terms: list[LessonTerm] = []
    seen: set[str] = set()
    for unit in batch:
        for term in _lexical_unit_known_terms(context.request, unit):
            if term.key not in seen:
                seen.add(term.key)
                known_terms.append(term)
    lesson = context.prose.lessons[batch[0].lesson_index]
    context_sentences = [
        lesson.title_sentence,
        *(sentence for block in lesson.blocks for sentence in block.sentences),
    ]
    instructions = [
        "This is a batch of the lexical-only stage. Return exactly one result in units for every "
        "supplied unit, each with its exact unit_id and frozen sentence key. Concatenating a "
        "unit's runs must reproduce its frozen_sentence.text byte-for-byte. Annotate every lexical "
        "token, including a title unit. Use null term_key only for whitespace and punctuation, "
        "isolate surrounding punctuation, and never translate, rewrite, or annotate grammar.",
        "Each unit result is independent: its terms catalog and term keys are local to that one "
        "unit. Define every referenced identity exactly once in that unit's terms and use short "
        "opaque keys such as term-1. Reuse a local key only for repeated occurrences of the exact "
        "same lemma, normalized POS, meaning, and dictionary pronunciation within that unit. "
        "Distinct senses stay distinct. The host owns cross-sentence identities and final keys.",
        "known_terms is a small relevant subset of established definitions. Reuse its key only "
        "for the exact same lexical identity and copy that definition exactly. Otherwise create a "
        "local opaque key; never derive keys from the surface, lemma, pronunciation, or meaning.",
        "context_sentences is the whole lesson for sense and lemma decisions. Output only the "
        "supplied units. Follow every language_guidance rule, including canonical lemmas, "
        "segmentation, POS normalization, and pronunciation formatting. Give every non-name term "
        "an approximate corpus frequency_rank; only proper names may use null.",
        TERM_IDENTITY_PREFLIGHT_INSTRUCTIONS,
    ]
    return GenerationLexicalBatchRequest(
        **stage_request_common(context.request, include_failures=False),
        units=[
            LexicalBatchUnit(
                unit_id=unit.unit_id,
                lesson_index=unit.lesson_index,
                sentence_index=unit.sentence_index,
                is_title=unit.is_title,
                frozen_sentence=unit.sentence,
            )
            for unit in batch
        ],
        context_sentences=context_sentences,
        known_terms=known_terms[:256],
        language_guidance=context.request.brief.language_guidance,
        instructions=" ".join(instructions),
    )


def _make_unit_invocation(
    context: LexicalStageContext,
    unit: LexicalUnit,
    *,
    unit_attempt: int,
    repair_result: GenerationLexicalUnitResult | None,
    repair_error: str | None,
) -> CallbackInvocation:
    suffix = "" if unit_attempt == 1 else "-repair"
    return make_stage_invocation(
        _lexical_unit_request(
            context.request,
            context.prose,
            unit,
            unit_attempt=unit_attempt,
            repair_result=repair_result,
            repair_error=repair_error,
        ),
        base_request=context.request,
        job_dir=context.manifest.job_dir,
        artifact_dir=context.manifest.response_path.parent,
        pipeline_state=context.state,
        artifact_stem=(f"lexical-unit-{unit.lesson_index:02d}-{unit.sentence_index:04d}{suffix}"),
    )


def run_lexical_conflict_chunks(
    callback: GenerationCallback,
    executor: ThreadPoolExecutor,
    base_request: GenerationCallbackRequest,
    groups: list[LexicalConflictGroup],
    *,
    job_dir: Path,
    artifact_dir: Path,
    state: PipelineState,
    log_path: Path,
    logs: list[str],
    policy: LexicalExecutionPolicy = DEFAULT_LEXICAL_EXECUTION_POLICY,
) -> tuple[GenerationLexicalConflictResult, ...]:
    """Resolve bounded lexical-identity groups with the shared repair queue."""

    chunks = [
        groups[index : index + policy.conflict_chunk_size]
        for index in range(0, len(groups), policy.conflict_chunk_size)
    ]
    jobs = [
        _RepairJob[tuple[int, list[LexicalConflictGroup]], GenerationLexicalConflictResult](
            (index, chunk)
        )
        for index, chunk in enumerate(chunks)
    ]
    results: dict[int, GenerationLexicalConflictResult] = {}

    def make_invocation(
        item: tuple[int, list[LexicalConflictGroup]],
        attempt: int,
        _repair: GenerationLexicalConflictResult | None,
        error: str | None,
    ) -> CallbackInvocation:
        chunk_index, chunk = item
        stem = (
            "lexical-conflicts"
            if len(chunks) == 1
            else f"lexical-conflicts-{chunk_index + 1:02d}-of-{len(chunks):02d}"
        )
        suffix = "" if attempt == 1 else "-repair"
        return make_stage_invocation(
            _lexical_conflict_request(base_request, chunk, repair_error=error),
            base_request=base_request,
            job_dir=job_dir,
            artifact_dir=artifact_dir,
            pipeline_state=state,
            artifact_stem=f"{stem}{suffix}",
        )

    def parse_result(
        _item: tuple[int, list[LexicalConflictGroup]],
        payload: str,
    ) -> GenerationLexicalConflictResult:
        return GenerationLexicalConflictResult.model_validate_json(payload)

    def validate_result(
        item: tuple[int, list[LexicalConflictGroup]],
        candidate: GenerationLexicalConflictResult,
    ) -> None:
        _chunk_index, chunk = item
        validate_lexical_conflict_result(chunk, candidate)

    def accept_result(
        item: tuple[int, list[LexicalConflictGroup]],
        result: GenerationLexicalConflictResult,
    ) -> None:
        chunk_index, _chunk = item
        results[chunk_index] = result

    failures = _run_repair_jobs(
        callback,
        executor,
        jobs,
        policy=policy,
        make_invocation=make_invocation,
        parse_result=parse_result,
        validate_result=validate_result,
        accept_result=accept_result,
        retain_failure=lambda _item, _repair: None,
        describe=lambda item: f"chunk {item[0] + 1}/{len(chunks)}",
        log_path=log_path,
        logs=logs,
    )
    if failures:
        raise ValueError(
            "identity conflict reconciliation failed after "
            f"{policy.local_repair_limit} local repair(s): " + "; ".join(failures)
        )
    if len(results) != len(chunks):
        raise ValueError("identity conflict reconciliation produced incomplete chunk results")
    return tuple(results[index] for index in range(len(chunks)))


def _write_lexical_partial(
    path: Path,
    prose: GenerationProseResult,
    request: GenerationCallbackRequest,
    units: Sequence[LexicalUnit],
    results: dict[str, GenerationLexicalUnitResult],
    repairs: dict[str, GenerationLexicalUnitResult],
) -> None:
    ordered_ids = [unit.unit_id for unit in units]
    payload = {
        "schema_version": 1,
        "profile_fingerprint": request.profile_fingerprint,
        "prose_sha256": prose_sha256(prose),
        "units": [
            results[unit_id].model_dump(mode="json")
            for unit_id in ordered_ids
            if unit_id in results
        ],
        "repairs": [
            repairs[unit_id].model_dump(mode="json")
            for unit_id in ordered_ids
            if unit_id in repairs and unit_id not in results
        ],
    }
    write_private(path, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _lexical_unit_request(
    base_request: GenerationCallbackRequest,
    prose: GenerationProseResult,
    unit: LexicalUnit,
    *,
    unit_attempt: int,
    repair_result: GenerationLexicalUnitResult | None,
    repair_error: str | None,
) -> GenerationLexicalUnitRequest:
    known_terms = _lexical_unit_known_terms(base_request, unit)
    instructions = [
        "This is one sentence of the lexical-only stage. Return exactly the supplied unit_id and "
        "frozen sentence key. Concatenating runs must reproduce frozen_sentence.text "
        "byte-for-byte. Annotate every lexical token, including the title when is_title is true. "
        "Use null term_key only for whitespace and punctuation, isolate surrounding punctuation, "
        "and never translate, rewrite, or annotate grammar.",
        "Term keys are local to this one response. Define every referenced identity exactly once "
        "in terms and use short opaque keys such as term-1. Reuse one local key for repeated "
        "occurrences of the exact same lemma, normalized POS, meaning, and dictionary "
        "pronunciation within this sentence. Distinct senses stay distinct. The host owns "
        "cross-sentence identities and final keys.",
        "known_terms is a small relevant subset of established definitions. Reuse its key only "
        "for the exact same lexical identity and copy that definition exactly. Otherwise create a "
        "local opaque key; never derive keys from the surface, lemma, pronunciation, or meaning.",
        "context_sentences supplies bounded reading context for sense and lemma decisions. Output "
        "only frozen_sentence. Follow every language_guidance rule, including canonical lemmas, "
        "segmentation, POS normalization, and pronunciation formatting. Give every non-name term "
        "an approximate corpus frequency_rank; only proper names may use null.",
        TERM_IDENTITY_PREFLIGHT_INSTRUCTIONS,
    ]
    if unit_attempt > 1:
        instructions.append(
            "This is the one allowed sentence-local repair. Correct the preceding output using "
            "the exact host diagnostic below. Do not change the frozen source."
        )
    elif repair_result is not None:
        instructions.append(
            "A preceding task attempt rejected only this sentence. Repair repair_result while "
            "keeping the frozen sentence authoritative."
        )
    if repair_error:
        instructions.append(f"Host diagnostic: {repair_error[:MAX_REPAIR_DIAGNOSTIC_CHARACTERS]}")
    return GenerationLexicalUnitRequest(
        **stage_request_common(base_request, include_failures=False),
        request_kind="sentence",
        unit_id=unit.unit_id,
        lesson_index=unit.lesson_index,
        sentence_index=unit.sentence_index,
        unit_attempt=unit_attempt,
        is_title=unit.is_title,
        frozen_sentence=unit.sentence,
        context_sentences=_lexical_unit_context(prose, unit),
        known_terms=known_terms,
        language_guidance=base_request.brief.language_guidance,
        repair_result=repair_result,
        instructions=" ".join(instructions),
    )


def _lexical_unit_context(
    prose: GenerationProseResult,
    unit: LexicalUnit,
) -> list[ProseLessonSentence]:
    lesson = prose.lessons[unit.lesson_index]
    sentences = [
        lesson.title_sentence,
        *(sentence for block in lesson.blocks for sentence in block.sentences),
    ]
    indices = sorted(
        {
            0,
            max(0, unit.sentence_index - 1),
            min(len(sentences) - 1, unit.sentence_index + 1),
        }
        - {unit.sentence_index}
    )
    return [sentences[index] for index in indices]


def _lexical_unit_known_terms(
    request: GenerationCallbackRequest,
    unit: LexicalUnit,
) -> list[LessonTerm]:
    priority = request.brief.priority_terms
    by_key = {term.key: term for term in priority}
    source = unit.sentence.text.casefold()
    ordered = [term for term in priority if term.lemma.casefold() in source]
    unmatched_candidates = (
        by_key[key]
        for key in request.target_policy.candidate_term_keys
        if key in by_key and by_key[key] not in ordered
    )
    ordered.extend(list(unmatched_candidates)[:MAX_UNMATCHED_PRIORITY_TERMS_PER_UNIT])
    selected: list[LessonTerm] = []
    seen: set[str] = set()
    for term in ordered:
        if term.key in seen:
            continue
        seen.add(term.key)
        selected.append(
            LessonTerm(
                key=term.key,
                lemma=term.lemma,
                pos=term.pos,
                gloss=term.gloss,
                pronunciation=term.pronunciation,
                frequency_rank=term.frequency_rank,
            )
        )
        if len(selected) >= MAX_KNOWN_TERMS_PER_UNIT:
            break
    return selected


def _lexical_conflict_request(
    base_request: GenerationCallbackRequest,
    groups: list[LexicalConflictGroup],
    *,
    repair_error: str | None = None,
) -> GenerationLexicalConflictRequest:
    instructions = [
        "Resolve only the supplied ambiguous lexical identity groups. Each group shares a "
        "normalized lemma but may differ in POS, gloss, or dictionary pronunciation. Contexts "
        "show actual frozen occurrences. Merge harmless POS-label variation only when the "
        "contextual lexical identity is genuinely the same; real homographs and distinct senses "
        "remain separate.",
        "Return exactly one resolution for every non-stored candidate. Map it to a candidate in "
        "the same group only when they are the same dictionary sense and pronunciation despite "
        "harmless wording variation. A stored candidate is an authoritative canonical anchor and "
        "must never itself be resolved. Different senses, readings, or homographs stay separate "
        "by mapping the generated candidate to itself. When uncertain, keep it separate.",
        "Do not create candidates, omit candidates, map across groups, or return chains. If a "
        "generated candidate is chosen as canonical, it must map to itself and every equivalent "
        "generated candidate maps directly to it.",
    ]
    if repair_error:
        instructions.append(
            "Repair the preceding conflict response using this host diagnostic: "
            f"{repair_error[:MAX_REPAIR_DIAGNOSTIC_CHARACTERS]}"
        )
    return GenerationLexicalConflictRequest(
        **stage_request_common(base_request, include_failures=False),
        request_kind="conflicts",
        groups=groups,
        instructions=" ".join(instructions),
    )
