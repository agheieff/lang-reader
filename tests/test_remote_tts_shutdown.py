"""Shutdown reuses Lang's token-fenced release without repeating uncertain uploads."""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest
from test_remote_tts import _task, _wave_bytes

from server.remote_tts import claim_remote_audio, complete_remote_audio, release_remote_audio
from server.remote_tts_worker import RemoteUnavailable, Worker
from server.workspaces import Workspace

pytest_plugins = ["test_remote_tts"]


def test_suspend_during_synthesis_releases_exact_claim_without_failure(
    workspace: Workspace,
) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    calls: list[str] = []

    class Remote:
        def claim(self) -> Any:
            calls.append("claim")
            return claim

        def release(self, profile: str, task_id: int, claim_token: str, reason: str) -> None:
            calls.append("release")
            assert claim_token == claim["claim_token"]
            release_remote_audio(workspace, task_id, reason, claim_token=claim_token)

    class Provider:
        closed = False

        def generate(self, request: Any) -> None:
            worker.stop(signal.SIGTERM, None)
            pytest.fail("Synthesis continued after stop")

        def close(self) -> None:
            self.closed = True

    provider = Provider()
    worker = Worker(Remote(), provider)  # type: ignore[arg-type]
    assert worker.step()
    assert provider.closed
    assert _task(workspace).state == "pending" and _task(workspace).attempts == 0
    assert not worker.step()
    assert len(calls) == 2


def test_stop_during_claim_releases_returned_token_before_synthesis(workspace: Workspace) -> None:
    class Remote:
        def claim(self) -> Any:
            worker.stop(signal.SIGTERM, None)
            self.taken = claim_remote_audio()
            return self.taken

        def release(self, profile: str, task_id: int, claim_token: str, reason: str) -> None:
            release_remote_audio(workspace, task_id, reason, claim_token=claim_token)

    class Provider:
        def generate(self, request: Any) -> None:
            pytest.fail("No synthesis after stopping")

        def close(self) -> None:
            pass

    worker = Worker(Remote(), Provider())  # type: ignore[arg-type]
    worker.step()
    assert _task(workspace).state == "pending" and _task(workspace).attempts == 0


def test_stop_during_uncertain_completion_does_not_release_or_fail(workspace: Workspace) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    calls: list[str] = []

    class Remote:
        def claim(self) -> Any:
            calls.append("claim")
            return claim

        def complete(self, profile: str, task_id: int, claim_token: str, audio: bytes) -> None:
            calls.append("complete")
            complete_remote_audio(workspace, task_id, audio, claim_token=claim_token)
            worker.stop(signal.SIGTERM, None)
            raise RemoteUnavailable("lost completion receipt")

    class Provider:
        def generate(self, request: Any) -> None:
            Path(request.output_path).write_bytes(_wave_bytes())

    worker = Worker(Remote(), Provider())  # type: ignore[arg-type]
    with pytest.raises(RemoteUnavailable):
        worker.step()
    assert len(calls) == 2
    assert _task(workspace).state == "completed"
    assert not worker.step()


def test_idle_and_unreachable_waits_release_the_provider_model() -> None:
    class Remote:
        def claim(self) -> Any:
            worker.stop(signal.SIGTERM, None)  # end the loop after one wait
            raise RemoteUnavailable("owner away")

    class Provider:
        closed = 0

        def close(self) -> None:
            self.closed += 1

    provider = Provider()
    worker = Worker(Remote(), provider)  # type: ignore[arg-type]
    worker.run()
    assert provider.closed == 1
