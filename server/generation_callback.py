"""Provider-neutral callback contracts, artifacts, and timing."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from server.clock import utc_now
from server.prompting import build_prompt
from server.schemas import (
    GenerationCallbackRequest,
    GenerationCallbackResult,
    GenerationGrammarRequest,
    GenerationGrammarResult,
    GenerationLexicalBatchRequest,
    GenerationLexicalBatchResult,
    GenerationLexicalConflictRequest,
    GenerationLexicalConflictResult,
    GenerationLexicalRequest,
    GenerationLexicalResult,
    GenerationLexicalUnitRequest,
    GenerationLexicalUnitResult,
    GenerationProseResult,
    GenerationTranslationRequest,
    GenerationTranslationResult,
    GrammarLessonDraft,
    PreferenceUpdateRequest,
    PreferenceUpdateResult,
    TranslationLessonDraft,
)

MAX_CALLBACK_BYTES = 5_000_000


class CallbackDeferred(RuntimeError):
    """The model command reported no capacity; this is not a failed draft."""


GenerationStageRequest = (
    GenerationCallbackRequest
    | GenerationLexicalUnitRequest
    | GenerationLexicalBatchRequest
    | GenerationLexicalConflictRequest
    | PreferenceUpdateRequest
    | GenerationLexicalRequest
    | GenerationTranslationRequest
    | GenerationGrammarRequest
)


@dataclass
class PipelineState:
    """Validated stage output retained while a generation task is assembled."""

    prose: GenerationProseResult | None = None
    lexical: GenerationLexicalResult | None = None
    translation: GenerationTranslationResult | None = None
    grammar: GenerationGrammarResult | None = None
    lexical_units: dict[str, GenerationLexicalUnitResult] = field(default_factory=dict)
    lexical_unit_repairs: dict[str, GenerationLexicalUnitResult] = field(default_factory=dict)
    translation_repair: list[TranslationLessonDraft] | None = None
    grammar_repair: list[GrammarLessonDraft] | None = None


@dataclass(frozen=True)
class CallbackInvocation:
    """One callback request plus its private artifact paths."""

    request: GenerationStageRequest
    job_dir: Path
    request_path: Path
    response_schema_path: Path
    response_path: Path
    base_request: GenerationCallbackRequest | None = None
    pipeline_state: PipelineState | None = None
    artifact_stem: str | None = None
    prompt_path: Path | None = None
    usage_path: Path | None = None


@dataclass(frozen=True)
class CallbackResult:
    """Raw callback output with host-recorded timing."""

    exit_code: int
    payload: str
    log: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    queue_wait_seconds: float | None = None
    deferred: bool = False
    usage: dict[str, int] = field(default_factory=dict)


class GenerationCallback(Protocol):
    """Replaceable provider adapter for one generation request."""

    def run(self, invocation: CallbackInvocation) -> CallbackResult: ...


def make_stage_invocation(
    request: GenerationStageRequest,
    *,
    base_request: GenerationCallbackRequest,
    job_dir: Path,
    artifact_dir: Path,
    pipeline_state: PipelineState | None = None,
    artifact_stem: str | None = None,
) -> CallbackInvocation:
    """Write one request and strict response schema, then describe its callback artifacts."""

    stem = artifact_stem or ("" if request.stage == "legacy" else request.stage)
    prefix = "" if not stem else f"{stem}-"
    request_path = artifact_dir / f"{prefix}request.json"
    schema_path = artifact_dir / f"{prefix}response-schema.json"
    response_path = artifact_dir / f"{prefix}response.json"
    prompt_path = artifact_dir / f"{prefix}prompt.txt"
    if isinstance(request, GenerationLexicalUnitRequest):
        result_type: type[Any] = GenerationLexicalUnitResult
    elif isinstance(request, GenerationLexicalBatchRequest):
        result_type = GenerationLexicalBatchResult
    elif isinstance(request, PreferenceUpdateRequest):
        result_type = PreferenceUpdateResult
    elif isinstance(request, GenerationLexicalConflictRequest):
        result_type = GenerationLexicalConflictResult
    else:
        result_type = {
            "legacy": GenerationCallbackResult,
            "prose": GenerationProseResult,
            "lexical": GenerationLexicalResult,
            "translation": GenerationTranslationResult,
            "grammar": GenerationGrammarResult,
        }[request.stage]
    write_private(request_path, canonical_request(request))
    write_private(prompt_path, build_prompt(request))
    schema = strict_output_schema(result_type.model_json_schema())
    if isinstance(request, GenerationGrammarRequest) and request.grammar_catalog:
        construction_schema = schema["$defs"]["CallbackGrammarOccurrence"]["properties"][
            "construction_key"
        ]
        construction_schema["enum"] = list(
            dict.fromkeys(construction.key for construction in request.grammar_catalog)
        )
    write_private(schema_path, json.dumps(schema, ensure_ascii=False, indent=2))
    return CallbackInvocation(
        request=request,
        job_dir=job_dir,
        request_path=request_path,
        response_schema_path=schema_path,
        response_path=response_path,
        base_request=base_request,
        pipeline_state=pipeline_state,
        artifact_stem=stem or None,
        prompt_path=prompt_path,
        usage_path=artifact_dir / f"{prefix}usage.json",
    )


def completed_stage_result(
    stage: str,
    future: Future[CallbackResult],
    invocation: CallbackInvocation,
    log_path: Path,
    logs: list[str],
) -> tuple[CallbackResult | None, str | None]:
    """Collect, record, and classify one externally executed callback."""

    try:
        result = future.result()
    except Exception as error:  # noqa: BLE001 - callback adapters are an external boundary
        return None, f"[{stage}] callback raised: {error}"
    record_stage_result(log_path, invocation, result, logs)
    result_error = callback_result_error(result)
    return result, None if result_error is None else f"[{stage}] {result_error}"


def record_stage_result(
    log_path: Path,
    invocation: CallbackInvocation,
    result: CallbackResult,
    logs: list[str],
) -> None:
    """Persist callback diagnostics, payload, and structured timing."""

    stage = invocation.request.stage
    artifact_stem = invocation.artifact_stem or stage
    timing_line = callback_timing_line(result)
    logs.append(f"[{artifact_stem}]\n{timing_line}\n{result.log}")
    write_private(log_path, "\n\n".join(logs))
    stage_log = invocation.response_path.parent / (
        "callback.log" if stage == "legacy" else f"{artifact_stem}-callback.log"
    )
    write_private(stage_log, f"{timing_line}\n{result.log}")
    event: dict[str, Any] = {
        "schema_version": 1,
        "kind": "callback",
        "artifact_stem": artifact_stem,
        "stage": stage,
        "task": invocation.request.task,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_seconds": rounded_seconds(result.duration_seconds),
        "queue_wait_seconds": rounded_seconds(result.queue_wait_seconds),
        "status": "deferred"
        if result.deferred
        else ("ok" if callback_result_error(result) is None else "error"),
        "exit_code": result.exit_code,
        "request_bytes": invocation.request_path.stat().st_size,
        "response_bytes": len(result.payload.encode()),
    }
    request_kind = getattr(invocation.request, "request_kind", None)
    if request_kind is not None:
        event["request_kind"] = request_kind
    if isinstance(invocation.request, GenerationLexicalUnitRequest):
        event.update(
            {
                "unit_id": invocation.request.unit_id,
                "unit_attempt": invocation.request.unit_attempt,
                "lesson_index": invocation.request.lesson_index,
                "sentence_index": invocation.request.sentence_index,
                "is_title": invocation.request.is_title,
                "source_characters": len(invocation.request.frozen_sentence.text),
            }
        )
    elif isinstance(invocation.request, GenerationLexicalBatchRequest):
        event.update(
            {
                "batch_units": len(invocation.request.units),
                "source_characters": sum(
                    len(unit.frozen_sentence.text) for unit in invocation.request.units
                ),
            }
        )
    elif isinstance(invocation.request, GenerationLexicalConflictRequest):
        event["conflict_groups"] = len(invocation.request.groups)
    event.update(result.usage)
    append_timing_event(log_path, event)
    if len(result.payload.encode()) <= MAX_CALLBACK_BYTES:
        write_private(invocation.response_path, result.payload)


def timing_start() -> tuple[str, float]:
    return utc_now().isoformat(), time.perf_counter()


def run_timed_callback(
    callback: GenerationCallback,
    invocation: CallbackInvocation,
    queued_at: float | None = None,
) -> CallbackResult:
    """Execute one callback and attach monotonic duration and queue wait."""

    started_at, started = timing_start()
    queue_wait = 0.0 if queued_at is None else max(0.0, started - queued_at)
    result = callback.run(invocation)
    return replace(
        result,
        started_at=started_at,
        finished_at=utc_now().isoformat(),
        duration_seconds=max(0.0, time.perf_counter() - started),
        queue_wait_seconds=queue_wait,
    )


def record_host_timing(
    log_path: Path,
    operation: str,
    started_at: str,
    started: float,
    *,
    outcome: str,
) -> None:
    append_timing_event(
        log_path,
        {
            "schema_version": 1,
            "kind": "host",
            "operation": operation,
            "started_at": started_at,
            "finished_at": utc_now().isoformat(),
            "duration_seconds": round(max(0.0, time.perf_counter() - started), 6),
            "status": outcome,
        },
    )


def append_timing_event(log_path: Path, event: dict[str, Any]) -> None:
    task_id, attempt = timing_task_identity(log_path)
    event = {"task_id": task_id, "attempt": attempt, **event}
    path = log_path.with_name("timings.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
    path.chmod(0o600)


def timing_task_identity(log_path: Path) -> tuple[int | None, int | None]:
    parts = log_path.parent.name.split("-")
    if len(parts) != 4 or parts[0] != "task" or parts[2] != "attempt":
        return None, None
    try:
        return int(parts[1]), int(parts[3])
    except ValueError:
        return None, None


def callback_timing_line(result: CallbackResult) -> str:
    duration = rounded_seconds(result.duration_seconds)
    queue_wait = rounded_seconds(result.queue_wait_seconds)
    return f"timing: duration_seconds={duration} queue_wait_seconds={queue_wait}"


def rounded_seconds(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def callback_result_error(result: CallbackResult) -> str | None:
    if result.deferred:
        raise CallbackDeferred("the model command is unavailable; waiting for capacity")
    if result.exit_code != 0:
        return f"callback exited with status {result.exit_code}"
    if len(result.payload.encode()) > MAX_CALLBACK_BYTES:
        return "callback response exceeded the byte limit"
    return None


def stage_request_common(
    base_request: GenerationCallbackRequest,
    *,
    include_failures: bool = True,
) -> dict[str, Any]:
    """Copy stable callback metadata into a derived stage request."""

    return {
        "job_id": base_request.job_id,
        "task_id": base_request.task_id,
        "attempt": base_request.attempt,
        "lesson_count": base_request.lesson_count,
        "profile_key": base_request.profile_key,
        "workspace_path": base_request.workspace_path,
        "profile_fingerprint": base_request.profile_fingerprint,
        "learning_language": base_request.brief.profile.learning_language,
        "translation_language": base_request.brief.profile.translation_language,
        "previous_failures": base_request.previous_failures if include_failures else [],
    }


def prose_sha256(prose: GenerationProseResult) -> str:
    return hashlib.sha256(prose.model_dump_json().encode()).hexdigest()


def write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def canonical_request(request: GenerationStageRequest) -> str:
    return json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def strict_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a Pydantic JSON schema acceptable to strict callback providers."""

    reference_annotations = {
        "$comment",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "$ref" in value:
                siblings = set(value) - {"$ref"}
                unsupported = siblings - reference_annotations
                if unsupported:
                    names = ", ".join(sorted(unsupported))
                    raise ValueError(f"output schema $ref has structural siblings: {names}")
                for annotation in siblings:
                    value.pop(annotation)
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["additionalProperties"] = False
                value["required"] = list(properties)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return schema
