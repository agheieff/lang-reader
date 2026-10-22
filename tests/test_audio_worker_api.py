"""The HTTP queue API and the worker-side client, driven against the real app in-process."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from test_remote_tts import _task, _wave_bytes

from server.audio_worker_api import router
from server.clock import utc_now
from server.db import session_scope
from server.models import AudioTask
from server.remote_tts_worker import HttpRemote, RemoteUnavailable
from server.workspaces import Workspace

pytest_plugins = ["test_remote_tts"]
TOKEN = "worker-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> Any:
    from server.main import app

    monkeypatch.setattr("server.audio_worker_api.registry", _registry_of(workspace))
    monkeypatch.setenv("ARC_LANG_AUDIO_WORKER_TOKEN", TOKEN)
    with TestClient(app) as test_client:
        yield test_client


def _registry_of(workspace: Workspace) -> Any:
    import server.remote_tts as remote_tts

    return remote_tts.registry


def test_the_api_is_off_without_a_token_and_needs_the_right_one(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server.main import app

    monkeypatch.setattr("server.audio_worker_api.registry", _registry_of(workspace))
    monkeypatch.delenv("ARC_LANG_AUDIO_WORKER_TOKEN", raising=False)
    with TestClient(app) as test_client:
        assert test_client.post("/api/audio-worker/claim", headers=AUTH).status_code == 404
        monkeypatch.setenv("ARC_LANG_AUDIO_WORKER_TOKEN", TOKEN)
        assert test_client.post("/api/audio-worker/claim").status_code == 401
        wrong = {"Authorization": "Bearer nope"}
        assert test_client.post("/api/audio-worker/claim", headers=wrong).status_code == 401
    assert router.prefix == "/api/audio-worker"


def test_claim_upload_and_complete_over_http(client: TestClient, workspace: Workspace) -> None:
    claim = client.post("/api/audio-worker/claim", headers=AUTH)
    assert claim.status_code == 200
    body = claim.json()
    assert body["profile_id"] == "es-es" and "output_path" not in body["request"]
    assert client.post("/api/audio-worker/claim", headers=AUTH).status_code == 204

    bad = client.post(
        f"/api/audio-worker/es-es/{body['task_id']}/complete",
        params={"claim_token": body["claim_token"]},
        content=b"not audio",
        headers=AUTH,
    )
    assert bad.status_code == 422 and _task(workspace).state == "running"

    done = client.post(
        f"/api/audio-worker/es-es/{body['task_id']}/complete",
        params={"claim_token": body["claim_token"]},
        content=_wave_bytes(),
        headers=AUTH,
    )
    assert done.status_code == 200
    assert _task(workspace).state == "completed"
    assert (workspace.directory / done.json()["relative_path"]).is_file()
    again = client.post(  # a lost response may be retried with the same bytes
        f"/api/audio-worker/es-es/{body['task_id']}/complete",
        params={"claim_token": body["claim_token"]},
        content=_wave_bytes(),
        headers=AUTH,
    )
    assert again.status_code == 200 and again.json() == done.json()


def test_a_reassigned_claim_is_refused_with_a_conflict(
    client: TestClient, workspace: Workspace
) -> None:
    old = client.post("/api/audio-worker/claim", headers=AUTH).json()
    with session_scope(workspace) as db:
        task = db.scalar(select(AudioTask))
        assert task is not None
        task.started_at = utc_now() - timedelta(hours=2)
    current = client.post("/api/audio-worker/claim", headers=AUTH).json()
    assert current["claim_token"] != old["claim_token"]
    path = f"/api/audio-worker/es-es/{old['task_id']}"
    stale = {"claim_token": old["claim_token"]}
    assert (
        client.post(f"{path}/fail", json={**stale, "error": "x"}, headers=AUTH).status_code == 409
    )
    assert (
        client.post(f"{path}/release", json={**stale, "reason": "x"}, headers=AUTH).status_code
        == 409
    )
    assert (
        client.post(
            f"{path}/complete", params=stale, content=_wave_bytes(), headers=AUTH
        ).status_code
        == 409
    )
    assert _task(workspace).state == "running"


def test_fail_and_release_use_the_attempt_budget_correctly(
    client: TestClient, workspace: Workspace
) -> None:
    claim = client.post("/api/audio-worker/claim", headers=AUTH).json()
    path = f"/api/audio-worker/es-es/{claim['task_id']}"
    token = claim["claim_token"]
    released = client.post(
        f"{path}/release", json={"claim_token": token, "reason": "stopping"}, headers=AUTH
    )
    assert released.json() == {"state": "pending"} and _task(workspace).attempts == 0
    claim = client.post("/api/audio-worker/claim", headers=AUTH).json()
    failed = client.post(
        f"{path}/fail", json={"claim_token": claim["claim_token"], "error": "boom"}, headers=AUTH
    )
    assert failed.json() == {"state": "pending"} and _task(workspace).attempts == 1


def test_the_worker_login_exemption_does_not_open_the_rest_of_the_app(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import FastAPI

    from server.access import Access
    from server.settings import AuthSettings

    app = FastAPI()
    app.middleware("http")(Access(AuthSettings("password", "x", "X-User", (), ("testserver",))))
    app.include_router(router)

    @app.get("/api/profiles")
    def profiles() -> list[str]:
        return []

    monkeypatch.setenv("ARC_LANG_AUDIO_WORKER_TOKEN", TOKEN)
    with TestClient(app) as test_client:
        assert test_client.get("/api/profiles").status_code == 401
        assert test_client.post("/api/audio-worker/claim").status_code == 401  # token, not login
        assert test_client.post("/api/audio-worker/claim", headers=AUTH).status_code in {200, 204}


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_http_remote_speaks_the_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, dict[str, str], bytes]] = []
    answers = [b"null", b'{"relative_path": "audio/x.wav"}', b'{"state": "pending"}']

    def fake_urlopen(request: Any, timeout: float) -> _Response:
        seen.append((request.full_url, dict(request.header_items()), request.data))
        return _Response(answers.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    remote = HttpRemote("https://owner.example/lang", TOKEN)
    assert remote.claim() is None
    remote.complete("es-es", 7, "2026-10-22T10:00:00+00:00", b"RIFF")
    remote.fail("es-es", 7, "tok", "boom")
    assert seen[0][0] == "https://owner.example/lang/api/audio-worker/claim"
    assert seen[1][0].startswith("https://owner.example/lang/api/audio-worker/es-es/7/complete?")
    assert "claim_token=2026-10-22T10%3A00%3A00%2B00%3A00" in seen[1][0]
    assert seen[1][2] == b"RIFF" and seen[1][1]["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(seen[2][2]) == {"claim_token": "tok", "error": "boom"}


def test_http_remote_separates_unreachable_owner_from_refusals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(request: Any, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 409, "Conflict", None, io.BytesIO(b"stale"))  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    remote = HttpRemote("https://owner.example", TOKEN)
    with pytest.raises(RuntimeError, match="HTTP 409: stale"):
        remote.fail("es-es", 1, "t", "x")

    def down(request: Any, timeout: float) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", down)
    with pytest.raises(RemoteUnavailable, match="connection refused"):
        remote.claim()

    def overloaded(request: Any, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 502, "Bad Gateway", None, io.BytesIO(b""))  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", overloaded)
    with pytest.raises(RemoteUnavailable):
        remote.claim()
