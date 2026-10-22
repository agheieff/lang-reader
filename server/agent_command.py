"""Run a generation request through an external command (a ``kind = "command"`` provider).

The command is started as an argv list without a shell:

* stdin carries the canonical request JSON;
* ``ARC_LANG_CALLBACK_PROMPT_PATH`` names a ready chat prompt (``prompting.build_prompt``) and
  ``ARC_LANG_CALLBACK_RESPONSE_SCHEMA_PATH`` the JSON Schema the answer must satisfy;
* the answer is JSON on stdout, or in the file ``ARC_LANG_CALLBACK_RESPONSE_PATH`` when present;
* the command may write token counts to ``ARC_LANG_CALLBACK_USAGE_PATH`` as a JSON object;
* exit status ``ARC_LANG_AGENT_UNAVAILABLE_EXIT`` (default 3) means "no capacity right now, no
  effect happened": the task waits instead of being charged a failed attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path

from server.generation_callback import (
    MAX_CALLBACK_BYTES,
    CallbackInvocation,
    CallbackResult,
    canonical_request,
)

KILL_GRACE_SECONDS = 10
_BASE_ENVIRONMENT = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR")
USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def run_command(
    command: list[str],
    *,
    invocation: CallbackInvocation,
    timeout_seconds: int,
    passthrough: tuple[str, ...] = (),
) -> CallbackResult:
    request_json = canonical_request(invocation.request)
    if invocation.usage_path is not None:
        invocation.usage_path.unlink(missing_ok=True)  # never report a previous call's counts
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in (*_BASE_ENVIRONMENT, *passthrough)
    }
    environment.update(_protocol_environment(invocation, request_json))
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=invocation.job_dir,
            env=environment,
            start_new_session=True,
        )
    except OSError as error:
        return CallbackResult(exit_code=1, payload="", log=str(error))
    try:
        stdout, stderr = process.communicate(request_json, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _stop_group(process)
        return CallbackResult(exit_code=1, payload="", log=f"timed out after {timeout_seconds}s")
    payload = stdout
    response_path = invocation.response_path
    if response_path.is_file():
        response_path.chmod(0o600)
        payload = response_path.read_text(encoding="utf-8")
    return CallbackResult(
        exit_code=process.returncode,
        payload=payload,
        log=f"stdout:\n{stdout[-MAX_CALLBACK_BYTES:]}\n\nstderr:\n{stderr[-MAX_CALLBACK_BYTES:]}",
        usage=read_usage(invocation.usage_path),
    )


def _protocol_environment(invocation: CallbackInvocation, request_json: str) -> dict[str, str]:
    request = invocation.request
    environment = {
        "ARC_LANG_PROTOCOL": "1",
        "ARC_LANG_JOB_ID": request.job_id,
        "ARC_LANG_REQUEST_SHA256": hashlib.sha256(request_json.encode()).hexdigest(),
        "ARC_LANG_CALLBACK_PROTOCOL": "4",
        "ARC_LANG_CALLBACK_STAGE": request.stage,
        "ARC_LANG_CALLBACK_REQUEST_PATH": str(invocation.request_path),
        "ARC_LANG_CALLBACK_RESPONSE_SCHEMA_PATH": str(invocation.response_schema_path),
        "ARC_LANG_CALLBACK_RESPONSE_PATH": str(invocation.response_path),
        "ARC_LANG_GENERATION_TASK_ID": str(request.task_id),
        "ARC_LANG_PROFILE": request.profile_key,
        "ARC_LANG_WORKSPACE": request.workspace_path,
    }
    if invocation.prompt_path is not None:
        environment["ARC_LANG_CALLBACK_PROMPT_PATH"] = str(invocation.prompt_path)
    if invocation.usage_path is not None:
        environment["ARC_LANG_CALLBACK_USAGE_PATH"] = str(invocation.usage_path)
    return environment


def _stop_group(process: subprocess.Popen[str]) -> None:
    """Terminate the command and any helper it started, then wait for them."""

    for sig, wait in ((signal.SIGTERM, KILL_GRACE_SECONDS), (signal.SIGKILL, None)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.communicate(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue


def read_usage(path: Path | None) -> dict[str, int]:
    """Token counts a command reported; absent or malformed telemetry is simply empty."""

    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        field: value
        for field in USAGE_FIELDS
        if type(value := raw.get(field)) is int and value >= 0
    }
