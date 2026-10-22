"""HTTP face of the remote audio queue: a worker on another machine claims, synthesizes, uploads.

Enabled only when ``ARC_LANG_AUDIO_WORKER_TOKEN`` is set; every call needs that bearer token. The
operations and their claim-token fencing are those of ``remote_tts`` (the same ones ``lang tts``
offers over ssh); this module only carries them over HTTP.
"""

from __future__ import annotations

import hmac
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from server import remote_tts, settings
from server.workspaces import Workspace, registry

router = APIRouter(prefix="/api/audio-worker")


def require_worker(authorization: Annotated[str | None, Header()] = None) -> None:
    token = settings.load().tts.worker_token
    if token is None:
        raise HTTPException(status_code=404, detail="The audio worker API is not enabled")
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(status_code=401, detail="Invalid worker token")


Worker = Annotated[None, Depends(require_worker)]


class Failure(BaseModel):
    claim_token: str = Field(min_length=1)
    error: str = Field(min_length=1, max_length=4_000)


class Release(BaseModel):
    claim_token: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=4_000)


def _workspace(profile_id: str) -> Workspace:
    try:
        return registry.resolve(profile_id)
    except (LookupError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def _fenced(error: ValueError) -> HTTPException:
    """A stale claim is a conflict the worker must drop; anything else is a bad upload."""
    stale = "no longer current" in str(error)
    return HTTPException(status_code=409 if stale else 422, detail=str(error))


@router.post("/claim")
async def claim(_: Worker, lease_minutes: float = remote_tts.DEFAULT_LEASE_MINUTES) -> Any:
    task = await run_in_threadpool(remote_tts.claim_remote_audio, lease_minutes)
    return task if task is not None else Response(status_code=204)


@router.post("/{profile_id}/{task_id}/complete")
async def complete(
    _: Worker, request: Request, profile_id: str, task_id: int, claim_token: str
) -> dict[str, str]:
    declared = int(request.headers.get("content-length") or 0)
    if declared > remote_tts.MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="remote audio is too large")
    audio = await request.body()
    workspace = _workspace(profile_id)
    try:
        path = await run_in_threadpool(
            remote_tts.complete_remote_audio, workspace, task_id, audio, claim_token=claim_token
        )
    except ValueError as error:
        raise _fenced(error) from error
    return {"relative_path": path}


@router.post("/{profile_id}/{task_id}/fail")
async def fail(_: Worker, profile_id: str, task_id: int, body: Failure) -> dict[str, str]:
    workspace = _workspace(profile_id)
    try:
        state = await run_in_threadpool(
            remote_tts.fail_remote_audio,
            workspace,
            task_id,
            body.error,
            claim_token=body.claim_token,
        )
    except ValueError as error:
        raise _fenced(error) from error
    return {"state": state}


@router.post("/{profile_id}/{task_id}/release")
async def release(_: Worker, profile_id: str, task_id: int, body: Release) -> dict[str, str]:
    workspace = _workspace(profile_id)
    try:
        state = await run_in_threadpool(
            remote_tts.release_remote_audio,
            workspace,
            task_id,
            body.reason,
            claim_token=body.claim_token,
        )
    except ValueError as error:
        raise _fenced(error) from error
    return {"state": state}
