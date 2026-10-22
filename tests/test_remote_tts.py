from __future__ import annotations

import io
import wave
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from server.clock import utc_now
from server.db import init_db, session_scope
from server.learning import import_lesson
from server.models import AudioTask
from server.profile_activation import ProfileActivationUpdate, activate_profile_settings
from server.remote_tts import claim_remote_audio, complete_remote_audio, fail_remote_audio
from server.tts import audio_status
from server.workspaces import Workspace, WorkspaceRegistry


def _wave_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x00\x01" * 2_400)
    return buffer.getvalue()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lesson_factory: Any) -> Workspace:
    registry = WorkspaceRegistry(tmp_path)
    monkeypatch.setattr("server.remote_tts.registry", registry)
    resolved = registry.resolve("es-es")
    init_db(resolved)
    with session_scope(resolved) as db:
        activate_profile_settings(db, ProfileActivationUpdate())
        import_lesson(db, lesson_factory(key="remote-one"))
    return resolved


def _task(workspace: Workspace) -> AudioTask:
    with session_scope(workspace) as db:
        task = db.scalar(select(AudioTask))
        assert task is not None
        db.expunge(task)
        return task


def test_remote_worker_claims_uploads_and_completes_audio(workspace: Workspace) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    assert claim["profile_id"] == "es-es"
    assert claim["request"]["chunks"] and "output_path" not in claim["request"]
    assert claim_remote_audio() is None  # one running task, nothing else pending

    relative = complete_remote_audio(
        workspace, claim["task_id"], _wave_bytes(), claim_token=claim["claim_token"]
    )

    task = _task(workspace)
    assert task.state == "completed" and task.relative_path == relative
    assert (workspace.directory / relative).is_file()


def test_invalid_audio_is_rejected_and_reported_failures_use_attempts(
    workspace: Workspace,
) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    with pytest.raises(ValueError):
        complete_remote_audio(
            workspace, claim["task_id"], b"not audio", claim_token=claim["claim_token"]
        )
    assert _task(workspace).state == "running"

    assert (
        fail_remote_audio(
            workspace, claim["task_id"], "synthesis crashed", claim_token=claim["claim_token"]
        )
        == "pending"
    )
    assert _task(workspace).attempts == 1


def test_expired_lease_requeues_without_charging_an_attempt(workspace: Workspace) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    with session_scope(workspace) as db:
        task = db.get(AudioTask, claim["task_id"])
        assert task is not None
        task.started_at = utc_now() - timedelta(hours=2)
        db.commit()

    again = claim_remote_audio(lease_minutes=45)

    assert again is not None and again["task_id"] == claim["task_id"]
    assert _task(workspace).attempts == 1


def test_remote_mode_reports_queued_instead_of_unavailable(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("server.tts.provider_command", lambda: None)
    with session_scope(workspace) as db:
        lesson_id = db.scalar(select(AudioTask.lesson_id))
        assert lesson_id is not None
        assert audio_status(db, workspace, lesson_id).state == "unavailable"
        monkeypatch.setenv("ARC_LANG_TTS_REMOTE", "1")
        status = audio_status(db, workspace, lesson_id)
    assert status.reason == "queued"
    assert "audio worker" in (status.message or "")


def test_remote_worker_sends_provider_valid_requests_and_uploads(workspace: Workspace) -> None:
    from qwen_provider import _validate_request

    from server.remote_tts_worker import Worker

    claim = claim_remote_audio()
    assert claim is not None
    calls: list[tuple[str, int, str, bytes]] = []

    class FakeRemote:
        def claim(self) -> Any:
            return claim

        def complete(self, profile: str, task_id: int, claim_token: str, audio: bytes) -> None:
            calls.append((profile, task_id, claim_token, audio))

    class FakeProvider:
        def generate(self, request: Any) -> dict[str, Any]:
            _validate_request(request.model_dump())  # the real provider's contract
            Path(request.output_path).write_bytes(_wave_bytes())
            return {"ok": True}

    assert Worker(FakeRemote(), FakeProvider()).step() is True  # type: ignore[arg-type]
    assert calls == [("es-es", claim["task_id"], claim["claim_token"], _wave_bytes())]


def test_runtime_failures_release_without_using_attempts(workspace: Workspace) -> None:
    from server.remote_tts_worker import RuntimeBroken, Worker

    claim = claim_remote_audio()
    assert claim is not None
    released: list[str] = []

    class FakeRemote:
        def claim(self) -> Any:
            return claim

        def release(self, profile: str, task_id: int, claim_token: str, reason: str) -> None:
            from server.remote_tts import release_remote_audio

            released.append(reason)
            release_remote_audio(workspace, task_id, reason, claim_token=claim_token)

    class BrokenProvider:
        def generate(self, request: Any) -> dict[str, Any]:
            raise RuntimeError("LocalEntryNotFoundError: model snapshot missing")

    with pytest.raises(RuntimeBroken):
        Worker(FakeRemote(), BrokenProvider()).step()  # type: ignore[arg-type]
    assert "LocalEntryNotFoundError" in released[0]
    task = _task(workspace)
    assert task.state == "pending" and task.attempts == 0


def test_failed_task_can_be_retried_with_a_fresh_budget(workspace: Workspace) -> None:
    from server.remote_tts import retry_failed_audio

    for _ in range(2):
        claim = claim_remote_audio()
        assert claim is not None
        fail_remote_audio(
            workspace, claim["task_id"], "synthesis crashed", claim_token=claim["claim_token"]
        )
    task = _task(workspace)
    assert task.state == "failed"

    assert retry_failed_audio(workspace, task.id) == "pending"
    assert _task(workspace).attempts == 0


@pytest.mark.parametrize("operation", ["complete", "fail", "release"])
def test_reclaimed_task_rejects_the_old_workers_result(
    workspace: Workspace, operation: str
) -> None:
    from server.remote_tts import release_remote_audio

    old = claim_remote_audio()
    assert old is not None
    with session_scope(workspace) as db:
        task = db.get(AudioTask, old["task_id"])
        assert task is not None
        task.started_at = utc_now() - timedelta(hours=2)
    current = claim_remote_audio()
    assert current is not None and current["task_id"] == old["task_id"]
    assert current["claim_token"] != old["claim_token"]
    with pytest.raises(ValueError, match="no longer current"):
        if operation == "complete":
            complete_remote_audio(
                workspace, old["task_id"], _wave_bytes(), claim_token=old["claim_token"]
            )
        elif operation == "fail":
            fail_remote_audio(
                workspace, old["task_id"], "late failure", claim_token=old["claim_token"]
            )
        else:
            release_remote_audio(
                workspace, old["task_id"], "late release", claim_token=old["claim_token"]
            )
    task = _task(workspace)
    assert task.state == "running" and task.attempts == 1 and task.error is None
    assert not list((workspace.directory / "audio").glob("*.wav"))
    complete_remote_audio(
        workspace, current["task_id"], _wave_bytes(), claim_token=current["claim_token"]
    )
    assert _task(workspace).state == "completed"


def test_completed_upload_retry_is_idempotent_but_different_bytes_are_rejected(
    workspace: Workspace,
) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    args = (workspace, claim["task_id"], _wave_bytes())
    first = complete_remote_audio(*args, claim_token=claim["claim_token"])
    assert complete_remote_audio(*args, claim_token=claim["claim_token"]) == first
    with pytest.raises(LookupError):
        complete_remote_audio(
            workspace, claim["task_id"], b"changed", claim_token=claim["claim_token"]
        )
    assert (workspace.directory / first).read_bytes() == _wave_bytes()


def test_parallel_remote_claims_do_not_claim_one_task_twice(workspace: Workspace) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: claim_remote_audio(), range(2)))
    assert sum(claim is not None for claim in claims) == 1
    assert _task(workspace).attempts == 1


@pytest.mark.parametrize("lease", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_lease_does_not_claim_work(workspace: Workspace, lease: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        claim_remote_audio(lease)
    assert _task(workspace).state == "pending"


def test_truncated_wave_does_not_complete_or_publish(workspace: Workspace) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    with pytest.raises(ValueError, match="truncated WAV"):
        complete_remote_audio(
            workspace, claim["task_id"], _wave_bytes()[:-10], claim_token=claim["claim_token"]
        )
    assert _task(workspace).state == "running"
    assert not list((workspace.directory / "audio").glob("*.wav"))


def test_remote_synthesis_timeout_fits_inside_lease() -> None:
    from server.remote_tts import DEFAULT_LEASE_MINUTES
    from server.remote_tts_worker import SSH_TIMEOUT_SECONDS, SYNTHESIS_TIMEOUT_SECONDS

    assert SYNTHESIS_TIMEOUT_SECONDS + 2 * SSH_TIMEOUT_SECONDS < DEFAULT_LEASE_MINUTES * 60


def test_uncertain_upload_never_reports_a_synthesis_failure(workspace: Workspace) -> None:
    from server.remote_tts_worker import RemoteUnavailable, Worker

    claim = claim_remote_audio()
    assert claim is not None
    commands: list[str] = []

    class LostReceiptRemote:
        def claim(self) -> Any:
            commands.append("claim")
            return claim

        def complete(self, profile: str, task_id: int, claim_token: str, audio: bytes) -> None:
            commands.append("complete")
            complete_remote_audio(workspace, task_id, audio, claim_token=claim_token)
            raise RemoteUnavailable("response lost after commit")

    class FakeProvider:
        def generate(self, request: Any) -> dict[str, Any]:
            Path(request.output_path).write_bytes(_wave_bytes())
            return {"ok": True}

    with pytest.raises(RemoteUnavailable):
        Worker(LostReceiptRemote(), FakeProvider()).step()  # type: ignore[arg-type]
    assert len(commands) == 2 and _task(workspace).state == "completed"
    assert claim_remote_audio() is None


@pytest.fixture
def opus_format(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    monkeypatch.setenv("ARC_LANG_AUDIO_FORMAT", "opus")


def _silence(seconds: float = 2.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x00\x00" * int(24_000 * seconds))
    return buffer.getvalue()


@pytest.mark.usefixtures("opus_format")
def test_remote_audio_is_published_compressed_and_repeat_uploads_stay_idempotent(
    workspace: Workspace,
) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    upload = _silence()
    args = (workspace, claim["task_id"], upload)

    first = complete_remote_audio(*args, claim_token=claim["claim_token"])

    assert first.endswith(".opus") and _task(workspace).relative_path == first
    assert (workspace.directory / first).stat().st_size < len(upload) // 4
    assert not list((workspace.directory / "audio").glob("*.wav"))
    assert complete_remote_audio(*args, claim_token=claim["claim_token"]) == first
    with pytest.raises(LookupError):
        complete_remote_audio(
            workspace, claim["task_id"], _silence(3.0), claim_token=claim["claim_token"]
        )


@pytest.mark.usefixtures("opus_format")
def test_the_published_file_is_served_with_its_own_media_type(workspace: Workspace) -> None:
    from server.audio_files import media_type
    from server.tts import safe_audio_path

    claim = claim_remote_audio()
    assert claim is not None
    relative = complete_remote_audio(
        workspace, claim["task_id"], _silence(), claim_token=claim["claim_token"]
    )
    assert media_type(safe_audio_path(workspace, relative)) == "audio/ogg; codecs=opus"
