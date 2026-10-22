"""Owner operations that let a worker on another machine synthesize lesson audio.

The worker claims one task at a time over HTTP or SSH, synthesizes locally, and returns the WAV
bytes. A claim is a lease: a task left running longer than the lease (the worker went off
mid-task) is requeued without charging an attempt. Synthesis failures reported by the worker do
use the attempt budget.
"""

from __future__ import annotations

import fcntl
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.audio_files import (
    configured_suffix,
    publish,
    require_encoder,
    upload_digest,
    validate_wave,
)
from server.clock import as_utc, utc_now
from server.db import session_scope
from server.models import AudioTask
from server.tts import (
    audio_output_paths,
    claim_audio_task,
    complete_audio_task,
    fail_audio_task,
    lesson_for_audio_task,
    provider_request,
    release_audio_task,
    retry_audio_task,
)
from server.workspaces import Workspace, registry

DEFAULT_LEASE_MINUTES = 45.0
MAX_AUDIO_BYTES = 200_000_000


@contextmanager
def _remote_session(workspace: Workspace) -> Iterator[Session]:
    # Serialize lease recovery and publication across separate SSH processes, including the
    # filesystem rename. The local TTS worker must not run on a remote-queue host.
    workspace.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (workspace.directory / "remote-audio.lock").open("a", encoding="utf-8") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with session_scope(workspace) as db:
            yield db


def _claim_token(task: AudioTask) -> str:
    if task.started_at is None:
        raise ValueError("audio task has no claim")
    return as_utc(task.started_at).isoformat()


def _claimed_task(db: Session, task_id: int, claim_token: str) -> AudioTask:
    task = db.get(AudioTask, task_id)
    if task is None or task.started_at is None or _claim_token(task) != claim_token:
        raise ValueError(f"audio claim is no longer current: {task_id}")
    return task


def claim_remote_audio(lease_minutes: float = DEFAULT_LEASE_MINUTES) -> dict[str, Any] | None:
    if not math.isfinite(lease_minutes) or lease_minutes <= 0:
        raise ValueError("audio lease must be finite and positive")
    require_encoder(configured_suffix())
    selected = registry.selected_id()
    workspaces = sorted(registry.list(), key=lambda item: item.profile_id != selected)
    for workspace in workspaces:
        with _remote_session(workspace) as db:
            cutoff = utc_now() - timedelta(minutes=lease_minutes)
            for stale in db.scalars(select(AudioTask).where(AudioTask.state == "running")):
                if stale.started_at is None or as_utc(stale.started_at) < cutoff:
                    # The worker went away mid-task; that is not the task's fault.
                    release_audio_task(db, stale.id, reason="remote audio lease expired")
            task = claim_audio_task(db)
            if task is None:
                continue
            lesson = lesson_for_audio_task(db, task)
            request = provider_request(task, lesson, Path("remote.wav"))
            return {
                "profile_id": workspace.profile_id,
                "task_id": task.id,
                "claim_token": _claim_token(task),
                "request": request.model_dump(mode="json", exclude={"output_path"}),
            }
    return None


def complete_remote_audio(
    workspace: Workspace, task_id: int, audio: bytes, *, claim_token: str
) -> str:
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("remote audio is too large")
    with _remote_session(workspace) as db:
        task = _claimed_task(db, task_id, claim_token)
        temporary, final_path, relative_path = audio_output_paths(workspace, task)
        if task.state == "completed" and _same_upload(final_path, audio):
            # The first SSH response may have been lost after publication.
            return relative_path
        if task.state != "running":
            raise LookupError(f"running audio task not found: {task_id}")
        try:
            with temporary.open("wb") as output:
                os.fchmod(output.fileno(), 0o600)
                output.write(audio)
            validate_wave(temporary)
            publish(temporary, final_path)
            if final_path.suffix != ".wav":
                _digest_path(final_path).write_text(upload_digest(audio), encoding="utf-8")
        finally:
            temporary.unlink(missing_ok=True)
        complete_audio_task(db, task_id, relative_path=relative_path)
    return relative_path


def _digest_path(final_path: Path) -> Path:
    return final_path.with_name(f"{final_path.name}.upload-sha256")


def _same_upload(final_path: Path, audio: bytes) -> bool:
    """Whether the published file came from exactly these bytes (compressed files keep a digest)."""
    if not final_path.is_file():
        return False
    if final_path.suffix == ".wav":
        return final_path.read_bytes() == audio
    digest = _digest_path(final_path)
    return digest.is_file() and digest.read_text(encoding="utf-8") == upload_digest(audio)


def fail_remote_audio(workspace: Workspace, task_id: int, error: str, *, claim_token: str) -> str:
    with _remote_session(workspace) as db:
        _claimed_task(db, task_id, claim_token)
        return fail_audio_task(db, task_id, error=f"remote: {error}").state


def release_remote_audio(
    workspace: Workspace, task_id: int, reason: str, *, claim_token: str
) -> str:
    with _remote_session(workspace) as db:
        _claimed_task(db, task_id, claim_token)
        return release_audio_task(db, task_id, reason=f"remote: {reason}").state


def retry_failed_audio(workspace: Workspace, task_id: int) -> str:
    with _remote_session(workspace) as db:
        return retry_audio_task(db, task_id).state
