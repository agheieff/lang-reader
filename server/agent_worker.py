"""Durable lesson-generation worker with replaceable local callbacks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from server import generation_callback as callback_runtime
from server import settings as settings_module
from server.agent_brief import (
    build_agent_brief,
    build_generation_target_policy,
    calibration_generation_brief,
)
from server.content_domains import record_domain_rejection
from server.content_planning import (
    SubjectDomainError,
    build_content_plan,
    subject_domain_violation,
)
from server.db import init_db, session_scope
from server.generation_callback import (
    MAX_CALLBACK_BYTES,
    CallbackInvocation,
    CallbackResult,
    GenerationCallback,
    GenerationStageRequest,
    PipelineState,
)
from server.generation_instructions import (
    MAX_GENERATED_DIFFICULTY_DRIFT,
    MIN_GENERATED_LENGTH_RATIO,
    _generation_instructions,
    _grammar_generation_policy,
    _grammar_stage_instructions,
    _lexical_stage_instructions,
    _prose_stage_instructions,
    _scaled_target_count,
    _translation_stage_instructions,
)
from server.generation_lexical_stage import (
    DEFAULT_LEXICAL_EXECUTION_POLICY,
    LexicalExecutionPolicy,
    LexicalStageContext,
    run_lexical_conflict_chunks,
    run_sentence_lexical_stage,
)
from server.generation_normalization import (
    callback_reference_errors,
    normalize_callback_lesson,
)
from server.generation_queue import (
    MAX_LESSONS_PER_GENERATION_TASK,
    claim_generation_task,
    complete_generation_task,
    defer_generation_task,
    fail_generation_task,
    generation_failure_context,
    maintain_generation_task,
    maintain_topic_generation_tasks,
    recover_running_generation_tasks,
    unread_generation_lesson_count,
)
from server.generation_tasks import (
    GenerationMode,
    generated_lesson_key,
    generation_queue_target,
    generation_task_kind,
    generation_task_mode,
    requested_topic,
    topic_request_id,
)
from server.learning import import_lesson
from server.learning_units import (
    generated_unit_errors,
    term_identity,
)
from server.lesson_content import lesson_document
from server.lexical_generation import (
    enumerate_lexical_units,
    validate_lexical_unit_result,
)
from server.model_callback import load_callback
from server.models import GenerationTask, Lesson
from server.preference_agent import maybe_update_preferences
from server.profile_activation import profile_is_active
from server.profile_fingerprint import profile_fingerprint
from server.profile_state import ensure_profile
from server.schemas import (
    CalibrationGenerationBrief,
    CallbackGrammarOccurrence,
    CallbackLessonBlock,
    CallbackLessonDraft,
    CallbackLessonSentence,
    GeneratedLessonDraft,
    GenerationCallbackRequest,
    GenerationCallbackResult,
    GenerationFailureContext,
    GenerationGrammarRequest,
    GenerationGrammarResult,
    GenerationLexicalRequest,
    GenerationLexicalResult,
    GenerationLexicalUnitResult,
    GenerationProseResult,
    GenerationRequestKind,
    GenerationTargetPolicy,
    GenerationTranslationRequest,
    GenerationTranslationResult,
    GrammarLessonDraft,
    GrammarOccurrence,
    GrammarSentenceDraft,
    LessonCalibration,
    LessonDocument,
    LessonTerm,
    LexicalLessonDraft,
    ProseLessonDraft,
    TokenizedLessonBlock,
    TokenizedLessonDraft,
    TokenizedLessonSentence,
    TranslationLessonDraft,
)
from server.session_feedback import latest_profile_feedback
from server.workspaces import DATA_DIR, Workspace, registry

_canonical_request = callback_runtime.canonical_request
_strict_output_schema = callback_runtime.strict_output_schema
_run_lexical_conflict_chunks = run_lexical_conflict_chunks

DEFAULT_CALLBACK_TIMEOUT_SECONDS = 600
MAX_PARALLEL_GENERATION_CALLBACKS = 3


def process_generation_task(
    workspace: Workspace, task: GenerationTask, callback: GenerationCallback
) -> None:
    task_started_at, task_started = callback_runtime.timing_start()
    prepare_started_at, prepare_started = callback_runtime.timing_start()
    log_path, invocation = _prepare_invocation(workspace, task)
    callback_runtime.record_host_timing(
        log_path,
        "prepare",
        prepare_started_at,
        prepare_started,
        outcome="completed",
    )
    logs: list[str] = []
    outcome = "failed"
    try:
        if invocation.pipeline_state is None:
            result = callback_runtime.run_timed_callback(callback, invocation)
            callback_runtime.record_stage_result(log_path, invocation, result, logs)
            error = callback_runtime.callback_result_error(result)
            if error is not None:
                _fail(workspace, task.id, log_path, f"[legacy] {error}")
                return
            try:
                response = GenerationCallbackResult.model_validate_json(result.payload)
            except ValidationError as error:
                _fail(workspace, task.id, log_path, f"[legacy] invalid callback response: {error}")
                return
            final_request = invocation.base_request or cast(
                GenerationCallbackRequest,
                invocation.request,
            )
            final_response_path = invocation.response_path
        else:
            staged = _run_staged_pipeline(
                workspace,
                task,
                callback,
                invocation,
                log_path,
                logs,
            )
            if staged is None:
                return
            response, final_request, final_response_path = staged

        try:
            import_started_at, import_started = callback_runtime.timing_start()
            _import_callback_lessons(
                workspace,
                task,
                response,
                final_request,
                source_path=final_response_path,
            )
            callback_runtime.record_host_timing(
                log_path,
                "import",
                import_started_at,
                import_started,
                outcome="completed",
            )
        except (LookupError, ValueError) as error:
            callback_runtime.record_host_timing(
                log_path,
                "import",
                import_started_at,
                import_started,
                outcome="failed",
            )
            prefix = "[assembly] " if invocation.pipeline_state is not None else "[legacy] "
            _fail(workspace, task.id, log_path, f"{prefix}{error}")
            return

        with session_scope(workspace) as db:
            complete_generation_task(db, task.id, log_path=str(log_path))
        outcome = "completed"
    except callback_runtime.CallbackDeferred as error:
        with session_scope(workspace) as db:
            defer_generation_task(db, task.id, log_path=str(log_path), reason=str(error))
        outcome = "deferred"
    finally:
        callback_runtime.record_host_timing(
            log_path,
            "task",
            task_started_at,
            task_started,
            outcome=outcome,
        )
        shutil.rmtree(invocation.job_dir, ignore_errors=True)


@contextmanager
def _preserve_stage_failures(failures: list[str]) -> Iterator[None]:
    try:
        yield
    except callback_runtime.CallbackDeferred:
        # A parallel capacity wait cannot forgive an already validated stage failure.
        # Let the normal failure path retain that error and charge its bounded retry.
        if not failures:
            raise


def _run_staged_pipeline(
    workspace: Workspace,
    task: GenerationTask,
    callback: GenerationCallback,
    initial_invocation: CallbackInvocation,
    log_path: Path,
    logs: list[str],
) -> tuple[GenerationCallbackResult, GenerationCallbackRequest, Path] | None:
    state = initial_invocation.pipeline_state
    base_request = initial_invocation.base_request
    if state is None or base_request is None:
        raise ValueError("staged callback invocation is missing pipeline state")

    artifact_dir = initial_invocation.response_path.parent
    _materialize_reused_stages(
        state,
        base_request=base_request,
        job_dir=initial_invocation.job_dir,
        artifact_dir=artifact_dir,
    )
    for stage, value in (
        ("prose", state.prose),
        ("lexical", state.lexical),
        ("translation", state.translation),
        ("grammar", state.grammar),
    ):
        if value is not None:
            logs.append(f"[{stage}]\nreused validated output from the preceding attempt")
    if logs:
        callback_runtime.write_private(log_path, "\n\n".join(logs))

    if state.prose is None:
        result = callback_runtime.run_timed_callback(callback, initial_invocation)
        callback_runtime.record_stage_result(log_path, initial_invocation, result, logs)
        error = callback_runtime.callback_result_error(result)
        if error is not None:
            _fail(workspace, task.id, log_path, f"[prose] {error}")
            return None
        try:
            state.prose = GenerationProseResult.model_validate_json(result.payload)
            _validate_prose_result(state.prose, base_request)
        except (ValidationError, ValueError) as prose_error:
            if isinstance(prose_error, SubjectDomainError):
                with session_scope(workspace) as db:
                    record_domain_rejection(db, task, prose_error)
            # A protocol-1 command callback may still return one complete result. Keep that
            # transition path, but all protocol-3 callbacks use the real task DAG.
            try:
                legacy = GenerationCallbackResult.model_validate_json(result.payload)
            except ValidationError:
                _fail(
                    workspace,
                    task.id,
                    log_path,
                    f"[prose] invalid callback response: {prose_error}",
                )
                return None
            final_path = initial_invocation.response_path.parent / "response.json"
            callback_runtime.write_private(final_path, result.payload)
            frozen = _freeze_callback_lessons(legacy.lessons)
            callback_runtime.write_private(
                initial_invocation.response_path,
                GenerationProseResult(schema_version=1, lessons=frozen).model_dump_json(),
            )
            return legacy, base_request, final_path

    prose = state.prose
    if prose is None:
        raise ValueError("prose stage completed without prose")

    translation_invocation = (
        callback_runtime.make_stage_invocation(
            _stage_request(
                base_request,
                "translation",
                prose=prose,
                repair_lessons=state.translation_repair,
            ),
            base_request=base_request,
            job_dir=initial_invocation.job_dir,
            artifact_dir=artifact_dir,
            pipeline_state=state,
        )
        if state.translation is None
        else None
    )
    failures: list[str] = []
    with (
        ThreadPoolExecutor(
            max_workers=MAX_PARALLEL_GENERATION_CALLBACKS,
            thread_name_prefix="lang-generation",
        ) as executor,
        _preserve_stage_failures(failures),
    ):
        translation_future = (
            executor.submit(callback_runtime.run_timed_callback, callback, translation_invocation)
            if translation_invocation is not None
            else None
        )

        if state.lexical is None:
            lexical_started_at, lexical_started = callback_runtime.timing_start()
            lexical_outcome = "failed"
            try:
                lexical_manifest = callback_runtime.make_stage_invocation(
                    _stage_request(base_request, "lexical", prose=prose),
                    base_request=base_request,
                    job_dir=initial_invocation.job_dir,
                    artifact_dir=artifact_dir,
                    pipeline_state=state,
                )
                state.lexical = run_sentence_lexical_stage(
                    callback,
                    executor,
                    LexicalStageContext(
                        task_id=task.id,
                        prose=prose,
                        request=base_request,
                        state=state,
                        manifest=lexical_manifest,
                        log_path=log_path,
                        logs=logs,
                        stored_terms=_running_task_term_definitions(workspace, task.id),
                    ),
                    policy=LexicalExecutionPolicy(
                        batch_size=settings_module.load().agent.lexical_batch_size
                        or DEFAULT_LEXICAL_EXECUTION_POLICY.batch_size
                    ),
                )
                _validate_lexical_result(
                    workspace,
                    task,
                    state.lexical,
                    prose,
                    base_request,
                )
                lexical_outcome = "completed"
            except _StageValidationError as error:
                failures.append(f"[{error.stage}] {error}")
                state.lexical = None
            except (ValidationError, ValueError) as error:
                failures.append(f"[lexical] invalid callback response: {error}")
                state.lexical = None
            finally:
                callback_runtime.record_host_timing(
                    log_path,
                    "lexical_stage",
                    lexical_started_at,
                    lexical_started,
                    outcome=lexical_outcome,
                )

        grammar_invocation: CallbackInvocation | None = None
        grammar_future: Future[CallbackResult] | None = None
        if state.lexical is not None and state.grammar is None:
            grammar_invocation = callback_runtime.make_stage_invocation(
                _stage_request(
                    base_request,
                    "grammar",
                    prose=prose,
                    lexical=state.lexical,
                    repair_lessons=state.grammar_repair,
                ),
                base_request=base_request,
                job_dir=initial_invocation.job_dir,
                artifact_dir=artifact_dir,
                pipeline_state=state,
            )
            grammar_future = executor.submit(
                callback_runtime.run_timed_callback, callback, grammar_invocation
            )

        if translation_future is not None and translation_invocation is not None:
            stage_result, callback_error = callback_runtime.completed_stage_result(
                "translation",
                translation_future,
                translation_invocation,
                log_path,
                logs,
            )
            if callback_error is not None:
                failures.append(callback_error)
            elif stage_result is not None:
                try:
                    state.translation = GenerationTranslationResult.model_validate_json(
                        stage_result.payload
                    )
                    _validate_translation_result(state.translation, prose)
                except (ValidationError, ValueError) as error:
                    failures.append(f"[translation] invalid callback response: {error}")
                    state.translation = None

        if grammar_future is not None and grammar_invocation is not None:
            stage_result, callback_error = callback_runtime.completed_stage_result(
                "grammar",
                grammar_future,
                grammar_invocation,
                log_path,
                logs,
            )
            if callback_error is not None:
                failures.append(callback_error)
            elif stage_result is not None:
                try:
                    state.grammar = GenerationGrammarResult.model_validate_json(
                        stage_result.payload
                    )
                    _validate_grammar_result(state.grammar, prose, state.lexical, base_request)
                except (ValidationError, ValueError) as error:
                    failures.append(f"[grammar] invalid callback response: {error}")
                    state.grammar = None

    if failures:
        _fail(workspace, task.id, log_path, "; ".join(failures))
        return None
    if state.lexical is None or state.translation is None or state.grammar is None:
        _fail(workspace, task.id, log_path, "[assembly] staged generation is incomplete")
        return None
    assembly_started_at, assembly_started = callback_runtime.timing_start()
    try:
        response = _merge_stage_results(
            prose,
            state.lexical,
            state.translation,
            state.grammar,
        )
        _validate_frozen_lesson_text(response, prose.lessons)
    except (ValidationError, ValueError) as error:
        callback_runtime.record_host_timing(
            log_path,
            "assembly",
            assembly_started_at,
            assembly_started,
            outcome="failed",
        )
        _fail(workspace, task.id, log_path, f"[assembly] invalid staged merge: {error}")
        return None
    response_path = artifact_dir / "response.json"
    payload = response.model_dump_json()
    if len(payload.encode()) > MAX_CALLBACK_BYTES:
        callback_runtime.record_host_timing(
            log_path,
            "assembly",
            assembly_started_at,
            assembly_started,
            outcome="failed",
        )
        _fail(workspace, task.id, log_path, "[assembly] merged response exceeded the byte limit")
        return None
    callback_runtime.write_private(response_path, payload)
    callback_runtime.record_host_timing(
        log_path,
        "assembly",
        assembly_started_at,
        assembly_started,
        outcome="completed",
    )
    return response, base_request, response_path


def _materialize_reused_stages(
    state: PipelineState,
    *,
    base_request: GenerationCallbackRequest,
    job_dir: Path,
    artifact_dir: Path,
) -> None:
    staged_results = (
        ("prose", state.prose),
        ("lexical", state.lexical),
        ("translation", state.translation),
        ("grammar", state.grammar),
    )
    for stage, result in staged_results:
        if result is None:
            continue
        request = _stage_request(
            base_request,
            stage,
            prose=state.prose,
            lexical=state.lexical,
        )
        invocation = callback_runtime.make_stage_invocation(
            request,
            base_request=base_request,
            job_dir=job_dir,
            artifact_dir=artifact_dir,
            pipeline_state=state,
        )
        callback_runtime.write_private(invocation.response_path, result.model_dump_json())


class _StageValidationError(ValueError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def _prepare_invocation(
    workspace: Workspace, task: GenerationTask
) -> tuple[Path, CallbackInvocation]:
    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, task.id)
        if stored is None or stored.state != "running":
            raise ValueError(f"generation task {task.id} is not running")
        profile = ensure_profile(db)
        if not profile_is_active(profile):
            raise ValueError("generation task belongs to an inactive language profile")
        kind = generation_task_kind(stored)
        fingerprint = profile_fingerprint(db, profile)
        expected = stored.payload.get("profile_fingerprint")
        if expected != fingerprint:
            if kind == "queue_fill":
                raise ValueError("generation task profile is stale")
            payload = dict(stored.payload)
            payload.update(
                {
                    "profile_fingerprint": fingerprint,
                    "learning_language": profile.learning_language,
                    "translation_language": profile.translation_language,
                }
            )
            stored.payload = payload
            db.commit()
        mode = generation_task_mode(stored)
        needed = _generation_step_needed(db, stored, mode)
        if needed <= 0:
            job_dir = _job_dir(workspace, stored)
            log_path = job_dir / "callback.log"
            callback_runtime.write_private(log_path, "queue target already satisfied\n")
            complete_generation_task(db, stored.id, log_path=str(log_path))
            raise _QueueSatisfied
        calibration = calibration_generation_brief(db) if mode == "calibration" else None
        if mode == "calibration" and calibration is None:
            raise ValueError("calibration task no longer matches the profile")
        base_request = _build_generation_request(
            db,
            workspace=workspace,
            task=stored,
            fingerprint=fingerprint,
            lesson_count=needed,
            kind=kind,
            mode=mode,
            calibration=calibration,
        )

    artifact_dir = _job_dir(workspace, task)
    isolated_dir = Path(
        tempfile.mkdtemp(prefix=f"lang-reader-{workspace.profile_id}-task-{task.id}-")
    )
    log_path = artifact_dir / "callback.log"
    if calibration is not None:
        invocation = callback_runtime.make_stage_invocation(
            base_request,
            base_request=base_request,
            job_dir=isolated_dir,
            artifact_dir=artifact_dir,
        )
    else:
        state = _previous_pipeline_state(
            workspace,
            task,
            base_request,
        )
        if state.prose is None:
            invocation = callback_runtime.make_stage_invocation(
                _stage_request(base_request, "prose"),
                base_request=base_request,
                job_dir=isolated_dir,
                artifact_dir=artifact_dir,
                pipeline_state=state,
            )
        else:
            invocation = CallbackInvocation(
                base_request,
                job_dir=isolated_dir,
                request_path=artifact_dir / "assembly-request.json",
                response_schema_path=artifact_dir / "assembly-response-schema.json",
                response_path=artifact_dir / "assembly-response.json",
                base_request=base_request,
                pipeline_state=state,
            )
    return log_path, invocation


def _stage_request(
    base_request: GenerationCallbackRequest,
    stage: str,
    *,
    prose: GenerationProseResult | None = None,
    lexical: GenerationLexicalResult | None = None,
    repair_lessons: Sequence[LexicalLessonDraft | TranslationLessonDraft | GrammarLessonDraft]
    | None = None,
) -> GenerationStageRequest:
    if stage == "prose":
        payload = base_request.model_dump(mode="python")
        payload.update(
            {
                "stage": "prose",
                "task": "prose",
                "repair_lessons": base_request.repair_lessons,
                "instructions": base_request.instructions,
            }
        )
        return GenerationCallbackRequest.model_validate(payload)
    if prose is None:
        raise ValueError(f"{stage} stage requires frozen prose")
    common = callback_runtime.stage_request_common(base_request)
    repairs = list(repair_lessons or ())
    if stage == "lexical":
        return GenerationLexicalRequest(
            **common,
            frozen_lessons=prose.lessons,
            known_terms=base_request.brief.priority_terms,
            target_policy=base_request.target_policy,
            language_guidance=base_request.brief.language_guidance,
            repair_lessons=repairs,
            instructions=_lexical_stage_instructions(
                base_request,
                has_repair=bool(repairs),
            ),
        )
    if stage == "translation":
        return GenerationTranslationRequest(
            **common,
            frozen_lessons=prose.lessons,
            repair_lessons=repairs,
            instructions=_translation_stage_instructions(
                base_request,
                has_repair=bool(repairs),
            ),
        )
    if stage == "grammar":
        if lexical is None:
            raise ValueError("grammar stage requires lexical runs")
        return GenerationGrammarRequest(
            **common,
            tokenized_lessons=_tokenized_lessons(prose, lexical),
            grammar_catalog=base_request.brief.grammar_catalog,
            repair_lessons=repairs,
            instructions=_grammar_stage_instructions(
                base_request,
                has_repair=bool(repairs),
            ),
        )
    raise ValueError(f"unsupported staged callback phase: {stage}")


def _previous_pipeline_state(
    workspace: Workspace,
    task: GenerationTask,
    base_request: GenerationCallbackRequest,
) -> PipelineState:
    state = PipelineState()
    if task.attempts <= 1:
        return state
    previous_dir = (
        workspace.directory / "agent" / "jobs" / f"task-{task.id}-attempt-{task.attempts - 1}"
    )
    failed_stage = _previous_failure_stage(task, base_request.previous_failures)
    if failed_stage == "prose":
        return state

    try:
        previous_request = GenerationCallbackRequest.model_validate_json(
            (previous_dir / "prose-request.json").read_text(encoding="utf-8")
        )
        prose = GenerationProseResult.model_validate_json(
            _bounded_stage_response(previous_dir / "prose-response.json")
        )
        _validate_prose_result(prose, previous_request)
    except (OSError, ValidationError, ValueError):
        return state
    if previous_request.stage != "prose" or not _same_prose_dependencies(
        previous_request, base_request
    ):
        return state
    state.prose = prose

    lexical_result: GenerationLexicalResult | None = None
    with suppress(OSError, ValidationError, ValueError):
        lexical_result = GenerationLexicalResult.model_validate_json(
            _bounded_stage_response(previous_dir / "lexical-response.json")
        )
    if failed_stage != "lexical" and lexical_result is not None:
        try:
            lexical_request = GenerationLexicalRequest.model_validate_json(
                (previous_dir / "lexical-request.json").read_text(encoding="utf-8")
            )
            _validate_reused_request(lexical_request, base_request)
            if lexical_request.frozen_lessons != prose.lessons:
                raise ValueError("reused lexical request has different frozen prose")
            _validate_lexical_result(
                workspace,
                task,
                lexical_result,
                prose,
                base_request,
            )
            state.lexical = lexical_result
        except _StageValidationError as error:
            if error.stage == "prose":
                return PipelineState()
        except (OSError, ValidationError, ValueError):
            pass
    if state.lexical is None:
        state.lexical_units, state.lexical_unit_repairs = _load_previous_lexical_partial(
            previous_dir,
            prose,
            base_request,
        )

    translation_result: GenerationTranslationResult | None = None
    try:
        translation_result = GenerationTranslationResult.model_validate_json(
            _bounded_stage_response(previous_dir / "translation-response.json")
        )
        state.translation_repair = translation_result.lessons
    except (OSError, ValidationError, ValueError):
        pass
    if failed_stage != "translation" and translation_result is not None:
        try:
            translation_request = GenerationTranslationRequest.model_validate_json(
                (previous_dir / "translation-request.json").read_text(encoding="utf-8")
            )
            _validate_reused_request(translation_request, base_request)
            if translation_request.frozen_lessons != prose.lessons:
                raise ValueError("reused translation request has different frozen prose")
            _validate_translation_result(translation_result, prose)
            state.translation = translation_result
        except (OSError, ValidationError, ValueError):
            pass

    grammar_result: GenerationGrammarResult | None = None
    try:
        grammar_result = GenerationGrammarResult.model_validate_json(
            _bounded_stage_response(previous_dir / "grammar-response.json")
        )
        state.grammar_repair = grammar_result.lessons
    except (OSError, ValidationError, ValueError):
        pass
    if failed_stage != "grammar" and grammar_result is not None and state.lexical is not None:
        try:
            grammar_request = GenerationGrammarRequest.model_validate_json(
                (previous_dir / "grammar-request.json").read_text(encoding="utf-8")
            )
            _validate_reused_request(grammar_request, base_request)
            if grammar_request.tokenized_lessons != _tokenized_lessons(prose, state.lexical):
                raise ValueError("reused grammar request has different lexical runs")
            _validate_grammar_result(grammar_result, prose, state.lexical, base_request)
            state.grammar = grammar_result
        except (OSError, ValidationError, ValueError):
            pass
    return state


def _same_prose_dependencies(
    previous: GenerationCallbackRequest,
    current: GenerationCallbackRequest,
) -> bool:
    """Whether frozen prose still represents the current adaptive generation snapshot."""

    return (
        previous.lesson_count == current.lesson_count
        and previous.profile_fingerprint == current.profile_fingerprint
        and previous.brief.model_dump(mode="json", exclude={"generated_at"})
        == current.brief.model_dump(mode="json", exclude={"generated_at"})
        and previous.target_policy == current.target_policy
        and previous.grammar_policy == current.grammar_policy
        and previous.content_plan == current.content_plan
        and previous.calibration == current.calibration
        and previous.request_kind == current.request_kind
        and previous.requested_topic == current.requested_topic
        and previous.latest_feedback == current.latest_feedback
    )


def _load_previous_lexical_partial(
    previous_dir: Path,
    prose: GenerationProseResult,
    request: GenerationCallbackRequest,
) -> tuple[
    dict[str, GenerationLexicalUnitResult],
    dict[str, GenerationLexicalUnitResult],
]:
    try:
        raw = json.loads(_bounded_stage_response(previous_dir / "lexical-partial.json"))
    except (json.JSONDecodeError, OSError, ValueError):
        return {}, {}
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 1
        or raw.get("profile_fingerprint") != request.profile_fingerprint
        or raw.get("prose_sha256") != callback_runtime.prose_sha256(prose)
    ):
        return {}, {}

    language = request.brief.profile.learning_language
    units = enumerate_lexical_units(prose, language)
    units_by_id = {unit.unit_id: unit for unit in units}

    def load_items(name: str, *, require_valid: bool) -> dict[str, GenerationLexicalUnitResult]:
        values = raw.get(name)
        if not isinstance(values, list):
            return {}
        loaded: dict[str, GenerationLexicalUnitResult] = {}
        for value in values:
            try:
                result = GenerationLexicalUnitResult.model_validate(value)
                unit = units_by_id[result.unit_id]
                if require_valid:
                    validate_lexical_unit_result(unit, result, language)
                elif result.key != unit.sentence.key:
                    raise ValueError("repair sentence key changed")
            except (KeyError, ValidationError, ValueError):
                continue
            if result.unit_id in loaded:
                return {}
            loaded[result.unit_id] = result
        return loaded

    valid = load_items("units", require_valid=True)
    repairs = {
        unit_id: result
        for unit_id, result in load_items("repairs", require_valid=False).items()
        if unit_id not in valid
    }
    return valid, repairs


def _previous_failure_stage(
    task: GenerationTask,
    failures: Sequence[GenerationFailureContext],
) -> str | None:
    for failure in reversed(failures):
        if failure.task_id != task.id:
            continue
        if failure.error.startswith("[") and "]" in failure.error:
            stage = failure.error[1 : failure.error.index("]")]
            if stage in {"prose", "lexical", "translation", "grammar", "assembly"}:
                return stage
        return None
    return None


def _bounded_stage_response(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > MAX_CALLBACK_BYTES:
        raise OSError(f"missing or oversized stage response: {path.name}")
    return path.read_text(encoding="utf-8")


def _validate_reused_request(
    request: GenerationLexicalRequest | GenerationTranslationRequest | GenerationGrammarRequest,
    base_request: GenerationCallbackRequest,
) -> None:
    if (
        request.profile_fingerprint != base_request.profile_fingerprint
        or request.lesson_count != base_request.lesson_count
    ):
        raise ValueError("reused stage request belongs to different profile state")


def _validate_prose_result(
    response: GenerationProseResult,
    request: GenerationCallbackRequest,
) -> None:
    if len(response.lessons) != request.lesson_count:
        raise ValueError(
            "prose callback returned the wrong lesson count: "
            f"got {len(response.lessons)}, requested {request.lesson_count}"
        )
    for index, lesson in enumerate(response.lessons, start=1):
        drift = abs(lesson.difficulty - request.brief.profile.difficulty)
        if drift > MAX_GENERATED_DIFFICULTY_DRIFT:
            raise ValueError(
                f"prose lesson {index} difficulty is too far from the requested target: "
                f"got {lesson.difficulty:.3f}, requested "
                f"{request.brief.profile.difficulty:.3f}, maximum drift "
                f"{MAX_GENERATED_DIFFICULTY_DRIFT:.2f}"
            )
    if request.content_plan is not None:
        violation = subject_domain_violation(
            request.content_plan, [lesson.subject_domain for lesson in response.lessons]
        )
        if violation is not None:
            raise violation


def _freeze_callback_lessons(
    lessons: Sequence[CallbackLessonDraft],
) -> list[ProseLessonDraft]:
    return [
        ProseLessonDraft.model_validate(
            {
                "title": lesson.title,
                "title_sentence": {
                    "key": lesson.title_sentence.key,
                    "text": "".join(run.text for run in lesson.title_sentence.runs),
                },
                "topic": lesson.topic,
                "content_angle": lesson.content_angle,
                "hypothesis": lesson.hypothesis,
                "subject_domain": lesson.subject_domain,
                "level": lesson.level,
                "difficulty": lesson.difficulty,
                "blocks": [
                    {
                        "key": block.key,
                        "sentences": [
                            {
                                "key": sentence.key,
                                "text": "".join(run.text for run in sentence.runs),
                            }
                            for sentence in block.sentences
                        ],
                    }
                    for block in lesson.blocks
                ],
            }
        )
        for lesson in lessons
    ]


def _tokenized_lessons(
    prose: GenerationProseResult,
    lexical: GenerationLexicalResult,
) -> list[TokenizedLessonDraft]:
    if len(prose.lessons) != len(lexical.lessons):
        raise ValueError("lexical lesson count does not match frozen prose")
    return [
        TokenizedLessonDraft(
            title_sentence=TokenizedLessonSentence(
                key=lexical_lesson.title_sentence.key,
                runs=[run.text for run in lexical_lesson.title_sentence.runs],
            ),
            blocks=[
                TokenizedLessonBlock(
                    key=block.key,
                    sentences=[
                        TokenizedLessonSentence(
                            key=sentence.key,
                            runs=[run.text for run in sentence.runs],
                        )
                        for sentence in block.sentences
                    ],
                )
                for block in lexical_lesson.blocks
            ],
        )
        for _prose_lesson, lexical_lesson in zip(
            prose.lessons,
            lexical.lessons,
            strict=True,
        )
    ]


def _validate_lexical_result(
    workspace: Workspace,
    task: GenerationTask,
    lexical: GenerationLexicalResult,
    prose: GenerationProseResult,
    request: GenerationCallbackRequest,
) -> None:
    response = _lexical_callback_response(prose, lexical)
    _validate_frozen_lesson_text(response, prose.lessons)
    for lexical_lesson in lexical.lessons:
        body_terms = {
            run.term_key
            for block in lexical_lesson.blocks
            for sentence in block.sentences
            for run in sentence.runs
            if run.term_key is not None
        }
        unknown_targets = set(lexical_lesson.target_term_keys) - body_terms
        if unknown_targets:
            names = ", ".join(sorted(unknown_targets))
            raise ValueError(f"lexical targets are absent from body text: {names}")

    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, task.id)
        if stored is None or stored.state != "running":
            raise ValueError(f"generation task {task.id} is not running")
        profile = ensure_profile(db)
        known_terms = _profile_term_definitions(
            db,
            learning_language=profile.learning_language,
            translation_language=profile.translation_language,
        )
        normalized = [
            normalize_callback_lesson(
                lesson,
                profile.learning_language,
                fallback_terms=known_terms,
            )
            for lesson in response.lessons
        ]
        reference_errors = [
            error for lesson in normalized for error in callback_reference_errors(lesson)
        ]
        if reference_errors:
            raise ValueError(
                "lexical output has unresolved term references: " + "; ".join(reference_errors)
            )
        for normalized_lesson in normalized:
            draft = normalized_lesson.expand()
            _canonicalize_known_term_definitions(
                db,
                draft,
                learning_language=profile.learning_language,
                translation_language=profile.translation_language,
                known_terms=known_terms,
            )
            unit_errors = generated_unit_errors(draft, profile.learning_language)
            if unit_errors:
                raise ValueError(
                    "lexical output has invalid learning units: " + "; ".join(unit_errors)
                )
            try:
                _validate_generated_lesson_quality(
                    draft,
                    mode="lesson",
                    target_difficulty=request.brief.profile.difficulty,
                    target_text_length=request.target_policy.target_text_length,
                )
            except ValueError as error:
                raise _StageValidationError("prose", str(error)) from error


def _lexical_callback_response(
    prose: GenerationProseResult,
    lexical: GenerationLexicalResult,
) -> GenerationCallbackResult:
    if len(prose.lessons) != len(lexical.lessons):
        raise ValueError("lexical lesson count does not match frozen prose")
    lessons: list[CallbackLessonDraft] = []
    for prose_lesson, lexical_lesson in zip(
        prose.lessons,
        lexical.lessons,
        strict=True,
    ):
        if len(prose_lesson.blocks) != len(lexical_lesson.blocks):
            raise ValueError("lexical block count does not match frozen prose")
        blocks: list[CallbackLessonBlock] = []
        for prose_block, lexical_block in zip(
            prose_lesson.blocks,
            lexical_lesson.blocks,
            strict=True,
        ):
            if prose_block.key != lexical_block.key:
                raise ValueError("lexical block key does not match frozen prose")
            if len(prose_block.sentences) != len(lexical_block.sentences):
                raise ValueError("lexical sentence count does not match frozen prose")
            blocks.append(
                CallbackLessonBlock(
                    key=lexical_block.key,
                    sentences=[
                        CallbackLessonSentence(
                            key=lexical_sentence.key,
                            runs=lexical_sentence.runs,
                            translation="",
                            grammar=[],
                        )
                        for lexical_sentence in lexical_block.sentences
                    ],
                )
            )
        lessons.append(
            CallbackLessonDraft(
                title=prose_lesson.title,
                terms=lexical_lesson.terms,
                title_sentence=CallbackLessonSentence(
                    key=lexical_lesson.title_sentence.key,
                    runs=lexical_lesson.title_sentence.runs,
                    translation="",
                    grammar=[],
                ),
                topic=prose_lesson.topic,
                content_angle=prose_lesson.content_angle,
                hypothesis=prose_lesson.hypothesis,
                subject_domain=prose_lesson.subject_domain,
                level=prose_lesson.level,
                difficulty=prose_lesson.difficulty,
                blocks=blocks,
                target_term_keys=lexical_lesson.target_term_keys,
                calibration=None,
            )
        )
    return GenerationCallbackResult(schema_version=1, lessons=lessons)


def _validate_translation_result(
    translation: GenerationTranslationResult,
    prose: GenerationProseResult,
) -> None:
    if len(translation.lessons) != len(prose.lessons):
        raise ValueError("translation lesson count does not match frozen prose")
    for lesson_index, (translated_lesson, prose_lesson) in enumerate(
        zip(translation.lessons, prose.lessons, strict=True),
        start=1,
    ):
        expected = _prose_sentence_keys(prose_lesson)
        actual = [sentence.key for sentence in translated_lesson.sentences]
        if actual != expected:
            raise ValueError(
                f"translation lesson {lesson_index} sentence order differs from frozen prose"
            )


def _validate_grammar_result(
    grammar: GenerationGrammarResult,
    prose: GenerationProseResult,
    lexical: GenerationLexicalResult | None,
    request: GenerationCallbackRequest,
) -> None:
    if lexical is None:
        raise ValueError("grammar validation requires lexical runs")
    if len(grammar.lessons) != len(prose.lessons) or len(lexical.lessons) != len(prose.lessons):
        raise ValueError("grammar lesson count does not match its dependencies")
    allowed = {construction.key for construction in request.brief.grammar_catalog}
    for lesson_index, (grammar_lesson, prose_lesson, lexical_lesson) in enumerate(
        zip(grammar.lessons, prose.lessons, lexical.lessons, strict=True),
        start=1,
    ):
        expected = _prose_sentence_keys(prose_lesson)
        actual = [sentence.key for sentence in grammar_lesson.sentences]
        if actual != expected:
            raise ValueError(
                f"grammar lesson {lesson_index} sentence order differs from frozen prose"
            )
        lexical_sentences = _lexical_sentences(lexical_lesson)
        for grammar_sentence, lexical_sentence in zip(
            grammar_lesson.sentences,
            lexical_sentences,
            strict=True,
        ):
            run_count = len(lexical_sentence.runs)
            normalized_occurrences: list[CallbackGrammarOccurrence] = []
            seen_occurrences: set[tuple[str, int, int]] = set()
            for occurrence in grammar_sentence.occurrences:
                if occurrence.construction_key not in allowed:
                    continue
                if occurrence.run_end == run_count + 1 and occurrence.run_start < run_count:
                    occurrence = occurrence.model_copy(update={"run_end": run_count})
                if occurrence.run_end > run_count:
                    # Grammar annotations are optional metadata over already validated prose and
                    # lexical runs. One-past is a common inclusive-end mistake that is safe to
                    # normalize; larger overflows have no unambiguous host-side repair, so omit
                    # only that occurrence instead of discarding the complete generated lesson.
                    continue
                identity = (
                    occurrence.construction_key,
                    occurrence.run_start,
                    occurrence.run_end,
                )
                if identity in seen_occurrences:
                    continue
                seen_occurrences.add(identity)
                normalized_occurrences.append(occurrence)
            grammar_sentence.occurrences = normalized_occurrences


def _merge_stage_results(
    prose: GenerationProseResult,
    lexical: GenerationLexicalResult,
    translation: GenerationTranslationResult,
    grammar: GenerationGrammarResult,
) -> GenerationCallbackResult:
    _validate_translation_result(translation, prose)
    if not (
        len(prose.lessons)
        == len(lexical.lessons)
        == len(translation.lessons)
        == len(grammar.lessons)
    ):
        raise ValueError("staged result lesson counts differ")
    lessons: list[CallbackLessonDraft] = []
    for prose_lesson, lexical_lesson, translation_lesson, grammar_lesson in zip(
        prose.lessons,
        lexical.lessons,
        translation.lessons,
        grammar.lessons,
        strict=True,
    ):
        translations = {
            sentence.key: sentence.translation for sentence in translation_lesson.sentences
        }
        grammar_by_key = {sentence.key: sentence for sentence in grammar_lesson.sentences}
        sentence_ordinal = 1
        title_key = prose_lesson.title_sentence.key
        title_grammar = _host_grammar_occurrences(
            grammar_by_key[title_key],
            sentence_ordinal=sentence_ordinal,
        )
        sentence_ordinal += 1
        blocks: list[CallbackLessonBlock] = []
        for prose_block, lexical_block in zip(
            prose_lesson.blocks,
            lexical_lesson.blocks,
            strict=True,
        ):
            sentences: list[CallbackLessonSentence] = []
            for prose_sentence, lexical_sentence in zip(
                prose_block.sentences,
                lexical_block.sentences,
                strict=True,
            ):
                key = prose_sentence.key
                sentences.append(
                    CallbackLessonSentence(
                        key=key,
                        runs=lexical_sentence.runs,
                        translation=translations[key],
                        grammar=_host_grammar_occurrences(
                            grammar_by_key[key],
                            sentence_ordinal=sentence_ordinal,
                        ),
                    )
                )
                sentence_ordinal += 1
            blocks.append(CallbackLessonBlock(key=prose_block.key, sentences=sentences))
        lessons.append(
            CallbackLessonDraft(
                title=prose_lesson.title,
                terms=lexical_lesson.terms,
                title_sentence=CallbackLessonSentence(
                    key=title_key,
                    runs=lexical_lesson.title_sentence.runs,
                    translation=translations[title_key],
                    grammar=title_grammar,
                ),
                topic=prose_lesson.topic,
                content_angle=prose_lesson.content_angle,
                hypothesis=prose_lesson.hypothesis,
                subject_domain=prose_lesson.subject_domain,
                level=prose_lesson.level,
                difficulty=prose_lesson.difficulty,
                blocks=blocks,
                target_term_keys=lexical_lesson.target_term_keys,
                calibration=None,
            )
        )
    return GenerationCallbackResult(schema_version=1, lessons=lessons)


def _host_grammar_occurrences(
    sentence: GrammarSentenceDraft,
    *,
    sentence_ordinal: int,
) -> list[GrammarOccurrence]:
    ordered = sorted(
        sentence.occurrences,
        key=lambda occurrence: (
            occurrence.run_start,
            occurrence.run_end,
            occurrence.construction_key,
            occurrence.note or "",
        ),
    )
    return [
        GrammarOccurrence(
            key=f"grammar:{sentence_ordinal}:{occurrence_ordinal}",
            construction_key=occurrence.construction_key,
            run_start=occurrence.run_start,
            run_end=occurrence.run_end,
            note=occurrence.note,
        )
        for occurrence_ordinal, occurrence in enumerate(ordered, start=1)
    ]


def _prose_sentence_keys(lesson: ProseLessonDraft) -> list[str]:
    return [
        lesson.title_sentence.key,
        *(sentence.key for block in lesson.blocks for sentence in block.sentences),
    ]


def _lexical_sentences(lesson: LexicalLessonDraft) -> list[Any]:
    return [
        lesson.title_sentence,
        *(sentence for block in lesson.blocks for sentence in block.sentences),
    ]


def _validate_frozen_lesson_text(
    response: GenerationCallbackResult,
    frozen_lessons: Sequence[ProseLessonDraft],
) -> None:
    if len(response.lessons) != len(frozen_lessons):
        raise ValueError(
            "staged output changed frozen lesson count: "
            f"got {len(response.lessons)}, expected {len(frozen_lessons)}"
        )
    for lesson_index, (lesson, frozen) in enumerate(
        zip(response.lessons, frozen_lessons, strict=True),
        start=1,
    ):
        if (
            lesson.title != frozen.title
            or lesson.topic != frozen.topic
            or lesson.level != frozen.level
            or lesson.difficulty != frozen.difficulty
        ):
            raise ValueError(f"staged output changed frozen lesson {lesson_index} metadata")
        title_text = "".join(run.text for run in lesson.title_sentence.runs)
        if (
            lesson.title_sentence.key != frozen.title_sentence.key
            or title_text != frozen.title_sentence.text
        ):
            raise ValueError(f"staged output changed frozen lesson {lesson_index} title")
        if len(lesson.blocks) != len(frozen.blocks):
            raise ValueError(f"staged output changed frozen lesson {lesson_index} block count")
        for block_index, (block, frozen_block) in enumerate(
            zip(lesson.blocks, frozen.blocks, strict=True),
            start=1,
        ):
            if block.key != frozen_block.key:
                raise ValueError(
                    f"staged output changed frozen lesson {lesson_index} block {block_index} key"
                )
            if len(block.sentences) != len(frozen_block.sentences):
                raise ValueError(
                    f"staged output changed frozen lesson {lesson_index} block {block_index} "
                    "sentence count"
                )
            for sentence_index, (sentence, frozen_sentence) in enumerate(
                zip(block.sentences, frozen_block.sentences, strict=True),
                start=1,
            ):
                text = "".join(run.text for run in sentence.runs)
                if sentence.key != frozen_sentence.key or text != frozen_sentence.text:
                    raise ValueError(
                        f"staged output changed frozen lesson {lesson_index} block {block_index} "
                        f"sentence {sentence_index}"
                    )


def _build_generation_request(
    db: Session,
    *,
    workspace: Workspace,
    task: GenerationTask,
    fingerprint: str,
    lesson_count: int,
    kind: GenerationRequestKind,
    mode: GenerationMode,
    calibration: CalibrationGenerationBrief | None,
) -> GenerationCallbackRequest:
    feedback = tuple(latest_profile_feedback(db))
    task.payload = {**task.payload, "latest_feedback": list(feedback)}
    db.commit()
    brief = build_agent_brief(db, feedback=feedback)
    failures = generation_failure_context(task)
    repair_lessons = _previous_rejected_lessons(workspace, task)
    target_policy = build_generation_target_policy(
        db,
        mode,
        brief=brief,
        feedback=feedback,
    )
    task_topic = requested_topic(task)
    content_plan = (
        None
        if calibration is not None
        else build_content_plan(
            brief,
            task_id=task.id,
            requested_topic=task_topic,
            feedback=feedback,
        )
    )
    grammar_policy = _grammar_generation_policy(
        brief,
        target_text_length=target_policy.target_text_length,
        calibration=calibration is not None,
        feedback=feedback,
    )
    request = GenerationCallbackRequest(
        job_id=_job_id(workspace, task.id),
        task_id=task.id,
        attempt=task.attempts,
        lesson_count=lesson_count,
        profile_key=workspace.profile_id,
        workspace_path=workspace.relative_directory,
        profile_fingerprint=fingerprint,
        brief=brief,
        target_policy=target_policy,
        grammar_policy=grammar_policy,
        content_plan=content_plan,
        calibration=calibration,
        request_kind=kind,
        requested_topic=task_topic,
        latest_feedback=list(feedback),
        previous_failures=failures,
        repair_lessons=repair_lessons,
        instructions=(
            _generation_instructions(
                brief=brief,
                target_policy=target_policy,
                grammar_policy=grammar_policy,
                calibration=calibration,
                content_plan=content_plan,
                kind=kind,
                requested_topic=task_topic,
                failures=failures,
                repair_lessons=repair_lessons,
            )
            if calibration is not None
            else "Staged prose instructions are initialized from this generation snapshot."
        ),
    )
    if calibration is None:
        request = request.model_copy(
            update={"instructions": _prose_stage_instructions(request)},
        )
    return request


def _previous_rejected_lessons(
    workspace: Workspace,
    task: GenerationTask,
) -> list[CallbackLessonDraft]:
    if task.attempts <= 1:
        return []
    previous_dir = (
        workspace.directory / "agent" / "jobs" / f"task-{task.id}-attempt-{task.attempts - 1}"
    )
    # Preserve repair context when the first attempt predates the current four-task contract.
    for filename in ("enrichment-response.json", "response.json"):
        path = previous_dir / filename
        if not path.is_file() or path.stat().st_size > MAX_CALLBACK_BYTES:
            continue
        try:
            response = GenerationCallbackResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            return response.lessons
        except (OSError, ValidationError):
            continue
    return []


class _QueueSatisfied(Exception):
    pass


def _import_callback_lessons(
    workspace: Workspace,
    task: GenerationTask,
    response: GenerationCallbackResult,
    request: GenerationCallbackRequest,
    *,
    source_path: Path,
) -> None:
    with session_scope(workspace) as db:
        stored = db.get(GenerationTask, task.id)
        if stored is None or stored.state != "running":
            raise ValueError(f"generation task {task.id} is not running")
        profile = ensure_profile(db)
        if stored.payload.get("profile_fingerprint") != profile_fingerprint(db, profile):
            raise ValueError("profile changed while the callback was running")
        known_terms = _profile_term_definitions(
            db,
            learning_language=profile.learning_language,
            translation_language=profile.translation_language,
        )
        normalized_lessons = [
            normalize_callback_lesson(
                lesson,
                profile.learning_language,
                fallback_terms=known_terms,
            )
            for lesson in response.lessons
        ]
        reference_errors = [
            error for lesson in normalized_lessons for error in callback_reference_errors(lesson)
        ]
        if reference_errors:
            raise ValueError(
                "callback returned unresolved term references: " + "; ".join(reference_errors)
            )
        drafts = [lesson.expand() for lesson in normalized_lessons]
        mode = generation_task_mode(stored)
        target_policy = request.target_policy
        grammar_policy = request.grammar_policy
        kind = generation_task_kind(stored)
        target = generation_queue_target(stored) if kind == "queue_fill" else None
        unread_before = unread_generation_lesson_count(db, mode)
        needed = _generation_step_needed(db, stored, mode)
        if len(drafts) < needed:
            raise ValueError("callback returned too few lessons")
        calibration = calibration_generation_brief(db) if mode == "calibration" else None
        for draft in drafts[:needed]:
            _canonicalize_known_term_definitions(
                db,
                draft,
                learning_language=profile.learning_language,
                translation_language=profile.translation_language,
                known_terms=known_terms,
            )
            unit_errors = generated_unit_errors(draft, profile.learning_language)
            if unit_errors:
                raise ValueError(
                    "callback returned invalid learning units: " + "; ".join(unit_errors)
                )
            _validate_generated_calibration(draft.calibration, mode, calibration)
            _validate_generated_lesson_quality(
                draft,
                mode=mode,
                target_difficulty=request.brief.profile.difficulty,
                target_text_length=target_policy.target_text_length,
            )

        candidate_indexes = [
            index
            for index in range(1, MAX_LESSONS_PER_GENERATION_TASK + 1)
            if db.scalar(
                select(Lesson.id).where(Lesson.key == generated_lesson_key(stored.id, index))
            )
            is None
        ]
        for draft, index in zip(drafts, candidate_indexes, strict=False):
            if (
                kind == "queue_fill"
                and target is not None
                and unread_generation_lesson_count(db, mode) >= target
            ):
                break
            target_term_keys = _reconcile_target_term_keys(draft, target_policy)
            annotated_grammar = _annotated_grammar_keys(draft)
            realized_grammar = [
                key for key in grammar_policy.offered_keys if key in annotated_grammar
            ]
            metadata: dict[str, Any] = {
                "generation_job_id": _job_id(workspace, stored.id),
                "generation_task_id": stored.id,
                "generation_index": index,
                "generation_mode": mode,
                "generation_request_kind": kind,
                "targets": {
                    "preferred": target_policy.preferred_count,
                    "max": target_policy.max_count,
                    "text_length": target_policy.target_text_length,
                    "allow_zero": target_policy.allow_zero,
                    "offered": target_policy.candidate_term_keys,
                    "advisory": draft.target_term_keys,
                    "realized": target_term_keys,
                },
                "grammar": {
                    "preferred": grammar_policy.preferred_count,
                    "max": grammar_policy.max_count,
                    "text_length": target_policy.target_text_length,
                    "allow_zero": True,
                    "offered": list(grammar_policy.offered_keys),
                    "realized": realized_grammar,
                    "annotated": annotated_grammar,
                },
            }
            if mode == "lesson":
                metadata.update(
                    {
                        "adaptive_target_difficulty": request.brief.profile.difficulty,
                        "generated_difficulty": draft.difficulty,
                    }
                )
            if request.content_plan is not None:
                metadata["content_plan"] = request.content_plan.model_dump(mode="json")
                if draft.content_angle:
                    metadata["content_angle"] = draft.content_angle
                if draft.subject_domain:
                    metadata["content_domain"] = draft.subject_domain
                if draft.hypothesis and request.content_plan.move in {"variation", "new"}:
                    metadata["content_hypothesis"] = draft.hypothesis
            task_topic = requested_topic(stored)
            if task_topic is not None:
                metadata.update(
                    {
                        "request_id": topic_request_id(stored),
                        "requested_topic": task_topic,
                        "generated_topic": draft.topic,
                    }
                )
            document = LessonDocument(
                key=generated_lesson_key(stored.id, index),
                title=draft.title,
                title_sentence=draft.title_sentence,
                learning_language=profile.learning_language,
                translation_language=profile.translation_language,
                topic=task_topic or draft.topic,
                level=request.brief.profile.level if mode == "lesson" else draft.level,
                difficulty=(
                    request.brief.profile.difficulty if mode == "lesson" else draft.difficulty
                ),
                blocks=draft.blocks,
                target_term_keys=target_term_keys,
                calibration=draft.calibration,
                metadata=metadata,
            )
            import_lesson(db, document, source_path=str(source_path))
        if kind == "topic_request":
            if not db.scalar(
                select(Lesson.id).where(Lesson.key == generated_lesson_key(stored.id, 1)).limit(1)
            ):
                raise ValueError("callback did not import the requested-topic lesson")
        else:
            if target is None:
                raise ValueError("queue-fill generation task has no target")
            expected = min(target, unread_before + needed)
            if unread_generation_lesson_count(db, mode) < expected:
                raise ValueError("callback did not restore the requested unread lesson queue step")


def _generation_step_needed(db: Session, task: GenerationTask, mode: GenerationMode) -> int:
    if generation_task_kind(task) == "topic_request":
        exists = db.scalar(
            select(Lesson.id).where(Lesson.key == generated_lesson_key(task.id, 1)).limit(1)
        )
        return 0 if exists is not None else 1
    target = generation_queue_target(task)
    shortfall = target - unread_generation_lesson_count(db, mode)
    return min(MAX_LESSONS_PER_GENERATION_TASK, max(0, shortfall))


def _canonicalize_known_term_definitions(
    db: Session,
    draft: GeneratedLessonDraft,
    *,
    learning_language: str,
    translation_language: str,
    known_terms: Sequence[LessonTerm] | None = None,
) -> int:
    """Restore stable definitions and isolate unambiguous generated-key collisions.

    A known key remains authoritative only when its normalized lemma matches exactly; this repairs
    callback drift in POS, gloss, pronunciation, or rank without fuzzy meaning or stemming. A
    different lemma must resolve by full exact identity or receive a new deterministic display key.
    """
    known_terms = (
        list(known_terms)
        if known_terms is not None
        else _profile_term_definitions(
            db,
            learning_language=learning_language,
            translation_language=translation_language,
        )
    )
    known_by_key: dict[str, LessonTerm] = {}
    known_by_identity: dict[tuple[str, str, str, str], LessonTerm] = {}
    for term in known_terms:
        known_by_key.setdefault(term.key, term)
        known_by_identity.setdefault(term_identity(term, learning_language), term)

    sentences = [
        draft.title_sentence,
        *(sentence for block in draft.blocks for sentence in block.sentences),
    ]
    draft_by_key: dict[str, LessonTerm] = {}
    for sentence in sentences:
        for run in sentence.runs:
            if run.term is not None:
                draft_by_key.setdefault(run.term.key, run.term)

    occupied_keys = set(known_by_key) | set(draft_by_key)
    resolved_by_key: dict[str, LessonTerm] = {}
    aliases: dict[str, str] = {}
    for key, term in sorted(draft_by_key.items()):
        identity = term_identity(term, learning_language)
        same_key = known_by_key.get(key)
        if same_key is not None and term_identity(same_key, learning_language)[0] == identity[0]:
            resolved = same_key
        elif (known_identity := known_by_identity.get(identity)) is not None:
            resolved = known_identity
        elif same_key is not None:
            collision_key = _definition_collision_key(key, identity, occupied_keys)
            occupied_keys.add(collision_key)
            resolved = term.model_copy(update={"key": collision_key})
        else:
            resolved = term
        resolved_by_key[key] = resolved
        if key != resolved.key:
            aliases[key] = resolved.key

    replacements = 0
    for sentence in sentences:
        for run in sentence.runs:
            if run.term is None:
                continue
            resolved = resolved_by_key[run.term.key]
            if resolved != run.term:
                run.term = resolved
                replacements += 1
    if aliases:
        draft.target_term_keys = list(
            dict.fromkeys(aliases.get(key, key) for key in draft.target_term_keys)
        )
        if draft.calibration is not None:
            rewritten = [
                aliases.get(probe.term_key, probe.term_key) for probe in draft.calibration.probes
            ]
            if len(rewritten) != len(set(rewritten)):
                raise ValueError("exact-definition aliases collapse calibration probe terms")
            for probe, term_key in zip(draft.calibration.probes, rewritten, strict=True):
                probe.term_key = term_key
    return replacements


def _profile_term_definitions(
    db: Session,
    *,
    learning_language: str,
    translation_language: str,
) -> list[LessonTerm]:
    lessons = db.scalars(
        select(Lesson)
        .where(
            Lesson.learning_language == learning_language,
            Lesson.translation_language == translation_language,
        )
        .order_by(Lesson.imported_at, Lesson.id)
    ).all()
    return [term for lesson in lessons for term in lesson_document(lesson).term_catalog().values()]


def _running_task_term_definitions(
    workspace: Workspace,
    task_id: int,
) -> list[LessonTerm]:
    with session_scope(workspace) as db:
        task = db.get(GenerationTask, task_id)
        if task is None or task.state != "running":
            raise ValueError(f"generation task {task_id} is not running")
        profile = ensure_profile(db)
        return _profile_term_definitions(
            db,
            learning_language=profile.learning_language,
            translation_language=profile.translation_language,
        )


def _definition_collision_key(
    original_key: str,
    identity: tuple[str, str, str, str],
    occupied_keys: set[str],
) -> str:
    material = json.dumps(
        [original_key, *identity],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(material.encode()).hexdigest()[:16]
    counter = 0
    while True:
        marker = digest if counter == 0 else f"{digest}-{counter}"
        suffix = f"~{marker}"
        candidate = f"{original_key[: 200 - len(suffix)]}{suffix}"
        if candidate not in occupied_keys:
            return candidate
        counter += 1


def _reconcile_target_term_keys(
    draft: GeneratedLessonDraft,
    policy: GenerationTargetPolicy,
) -> list[str]:
    present = {
        run.term.key
        for block in draft.blocks
        for sentence in block.sentences
        for run in sentence.runs
        if run.term is not None
    }
    lexical_run_count = sum(
        run.term is not None
        for block in draft.blocks
        for sentence in block.sentences
        for run in sentence.runs
    )
    if lexical_run_count == 0 or policy.max_count == 0:
        return []

    target_cap = _scaled_target_count(
        policy.max_count,
        actual_length=lexical_run_count,
        target_length=policy.target_text_length,
    )
    offered_present = [key for key in policy.candidate_term_keys if key in present]
    declared = set(draft.target_term_keys)
    realized = [key for key in offered_present if key in declared][:target_cap]

    fill_count = min(
        _scaled_target_count(
            policy.preferred_count,
            actual_length=lexical_run_count,
            target_length=policy.target_text_length,
        ),
        target_cap,
    )
    for key in offered_present:
        if len(realized) >= fill_count:
            break
        if key not in realized:
            realized.append(key)
    return realized


def _annotated_grammar_keys(draft: GeneratedLessonDraft) -> list[str]:
    sentences = [
        draft.title_sentence,
        *(sentence for block in draft.blocks for sentence in block.sentences),
    ]
    return list(
        dict.fromkeys(
            occurrence.construction_key for sentence in sentences for occurrence in sentence.grammar
        )
    )


def _validate_generated_lesson_quality(
    draft: GeneratedLessonDraft,
    *,
    mode: GenerationMode,
    target_difficulty: float,
    target_text_length: int,
) -> None:
    """Reject materially off-contract ordinary prose before it enters the lesson library."""

    if mode == "calibration":
        return
    drift = abs(draft.difficulty - target_difficulty)
    if drift > MAX_GENERATED_DIFFICULTY_DRIFT:
        raise ValueError(
            "callback lesson difficulty is too far from the requested target: "
            f"got {draft.difficulty:.3f}, requested {target_difficulty:.3f}, "
            f"maximum drift {MAX_GENERATED_DIFFICULTY_DRIFT:.2f}"
        )
    lexical_tokens = sum(
        run.term is not None
        for block in draft.blocks
        for sentence in block.sentences
        for run in sentence.runs
    )
    minimum_length = math.ceil(target_text_length * MIN_GENERATED_LENGTH_RATIO)
    if lexical_tokens < minimum_length:
        raise ValueError(
            "callback lesson is drastically shorter than the requested lexical length: "
            f"got {lexical_tokens} body lexical tokens, requested {target_text_length}, "
            f"minimum {minimum_length}"
        )


def _validate_generated_calibration(
    generated: LessonCalibration | None,
    mode: GenerationMode,
    requested: CalibrationGenerationBrief | None,
) -> None:
    if mode == "lesson":
        if generated is not None:
            raise ValueError("ordinary generation must not return calibration probes")
        return
    if generated is None or requested is None:
        raise ValueError("calibration generation must return calibration probes")
    if generated.sequence != requested.sequence:
        raise ValueError("callback returned the wrong calibration sequence")
    generated_difficulties = sorted(round(probe.difficulty, 6) for probe in generated.probes)
    requested_difficulties = sorted(round(value, 6) for value in requested.probe_difficulties)
    if generated_difficulties != requested_difficulties:
        raise ValueError("callback returned different calibration probe difficulties")
    excluded = set(requested.excluded_term_keys)
    if any(probe.term_key in excluded for probe in generated.probes):
        raise ValueError("callback reused an excluded calibration probe term")


def _fail(workspace: Workspace, task_id: int, log_path: Path, error: str) -> None:
    with session_scope(workspace) as db:
        fail_generation_task(db, task_id, log_path=str(log_path), error=error)


def _job_id(workspace: Workspace, task_id: int) -> str:
    return f"{workspace.profile_id}:{task_id}"


def _job_dir(workspace: Workspace, task: GenerationTask) -> Path:
    agent_directory = workspace.directory / "agent"
    agent_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    agent_directory.chmod(0o700)
    parent = agent_directory / "jobs"
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent.chmod(0o700)
    path = parent / f"task-{task.id}-attempt-{task.attempts}"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def _refresh_worker_workspaces(profile_id: str | None, known_profiles: set[str]) -> list[Workspace]:
    workspaces = [registry.resolve(profile_id)] if profile_id else registry.list()
    current_ids = {workspace.profile_id for workspace in workspaces}
    known_profiles.intersection_update(current_ids)
    for workspace in workspaces:
        if workspace.profile_id in known_profiles:
            continue
        init_db(workspace)
        with session_scope(workspace) as db:
            recover_running_generation_tasks(db)
            maintain_generation_task(db)
            maintain_topic_generation_tasks(db)
        known_profiles.add(workspace.profile_id)
    return workspaces


_host_admission_cache: tuple[str, float, bool] | None = None


def deferred_by_host() -> bool:
    """A configured admission command must exit zero to allow app-owned model work.

    Errors and unavailable allowance defer before claiming a task. A short observation
    cache bounds account inspection while the worker continues its ordinary polling.
    Local deployments without a host policy command retain their existing behavior.
    """
    global _host_admission_cache
    command = settings_module.load().agent.admission_command
    if not command:
        _host_admission_cache = None
        return False
    now = time.monotonic()
    if _host_admission_cache is not None:
        cached_command, checked, deferred = _host_admission_cache
        if cached_command == command and 0 <= now - checked < 60:
            return deferred
    try:
        deferred = (
            subprocess.run(shlex.split(command), capture_output=True, timeout=30).returncode != 0
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        deferred = True
    _host_admission_cache = (command, time.monotonic(), deferred)
    return deferred


def _maintain_worker_workspaces(
    workspaces: list[Workspace], callback: GenerationCallback | None = None
) -> None:
    for workspace in workspaces:
        with session_scope(workspace) as db:
            maintain_generation_task(db)
            maintain_topic_generation_tasks(db)
        if callback is not None and not deferred_by_host():
            maybe_update_preferences(workspace, callback)


def _round_robin_workspaces(
    workspaces: list[Workspace],
    last_profile_id: str | None,
    preferred_profile_id: str | None = None,
) -> list[Workspace]:
    ordered = workspaces
    if workspaces and last_profile_id is not None:
        for index, workspace in enumerate(workspaces):
            if workspace.profile_id == last_profile_id:
                start = (index + 1) % len(workspaces)
                ordered = workspaces[start:] + workspaces[:start]
                break
    if preferred_profile_id is None:
        return ordered
    preferred = [item for item in ordered if item.profile_id == preferred_profile_id]
    return preferred + [item for item in ordered if item.profile_id != preferred_profile_id]


def run_worker(*, once: bool = False, profile_id: str | None = None) -> None:
    callback = load_callback()
    lock_path = DATA_DIR / "agent" / "worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path.parent.chmod(0o700)
    with lock_path.open("w", encoding="utf-8") as lock:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another generation worker is already running") from error
        agent_settings = settings_module.load().agent
        poll_seconds = agent_settings.poll_seconds
        maintenance_seconds = agent_settings.maintenance_seconds
        next_maintenance = time.monotonic() + maintenance_seconds
        known_profiles: set[str] = set()
        last_profile_id: str | None = None
        while True:
            workspaces = _refresh_worker_workspaces(profile_id, known_profiles)
            now = time.monotonic()
            if now >= next_maintenance:
                _maintain_worker_workspaces(workspaces, callback)
                next_maintenance = now + maintenance_seconds
            if deferred_by_host():
                if once:
                    return
                time.sleep(poll_seconds)
                continue
            claimed: tuple[Workspace, GenerationTask] | None = None
            preferred_profile_id = profile_id or registry.selected_id()
            for workspace in _round_robin_workspaces(
                workspaces,
                last_profile_id,
                preferred_profile_id,
            ):
                with session_scope(workspace) as db:
                    task = claim_generation_task(db)
                if task is not None:
                    claimed = workspace, task
                    last_profile_id = workspace.profile_id
                    break
            if claimed is None:
                if once:
                    return
                time.sleep(poll_seconds)
                continue
            workspace, task = claimed
            try:
                process_generation_task(workspace, task, callback)
            except _QueueSatisfied:
                pass
            except (OSError, ValueError) as error:
                _fail(
                    workspace,
                    task.id,
                    _job_dir(workspace, task) / "callback.log",
                    str(error),
                )
            _maintain_worker_workspaces([workspace])
            if once:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local lesson generation worker")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--profile")
    args = parser.parse_args()
    try:
        run_worker(once=args.once, profile_id=args.profile)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
