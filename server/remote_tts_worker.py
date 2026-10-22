"""Audio worker for a remote queue: claims lesson audio from its owner and synthesizes it locally.

The queue owner (the host running the web service) keeps the tasks and the finished files; this
process reaches it over HTTP (``ARC_LANG_REMOTE_URL`` and a bearer token) or ssh, and needs the
local Qwen runtime (extras/tts-qwen). It runs whenever
this machine is on and backs off quietly while the owner is unreachable.
"""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from server import settings
from server.audio_files import validate_wave
from server.tts import AudioProviderRequest, provider_command
from server.tts_worker import ProviderProcess

IDLE_SECONDS = 120.0
UNREACHABLE_SECONDS = 300.0
RUNTIME_BROKEN_SECONDS = 1800.0
# Failures of this machine's runtime rather than of the task: release without using an attempt.
ENVIRONMENT_ERROR_MARKERS = (
    "LocalEntryNotFoundError",
    "GPU runtime is unavailable",
    "No module named",
    "Qwen provider stopped unexpectedly",
)
SSH_TIMEOUT_SECONDS = 120.0
HTTP_TIMEOUT_SECONDS = 300.0
# Finish or report failure before the owner's 45-minute lease can be reassigned.
SYNTHESIS_TIMEOUT_SECONDS = 40 * 60.0


class RemoteUnavailable(RuntimeError):
    pass


class ShutdownRequested(BaseException):
    """Stop synthesis without charging the owner a task failure."""


class RuntimeBroken(RuntimeError):
    pass


class Remote(Protocol):
    """The owner of the queue, however it is reached."""

    def claim(self) -> dict[str, Any] | None: ...

    def complete(self, profile: str, task_id: int, claim_token: str, audio: bytes) -> None: ...

    def fail(self, profile: str, task_id: int, claim_token: str, error: str) -> None: ...

    def release(self, profile: str, task_id: int, claim_token: str, reason: str) -> None: ...


class SshRemote:
    """The owner's ``lang tts`` commands over ssh."""

    def __init__(self, host: str, cli: str) -> None:
        self.host = host
        self.cli = cli

    def _run(self, arguments: list[str], *, stdin: bytes | None = None) -> Any:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            self.host,
            f"{self.cli} {shlex.join(arguments)}",
        ]
        try:
            completed = subprocess.run(
                command, input=stdin, capture_output=True, timeout=SSH_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as error:
            raise RemoteUnavailable("ssh timed out") from error
        if completed.returncode == 255:
            raise RemoteUnavailable(completed.stderr.decode(errors="replace").strip())
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode(errors="replace").strip())
        return json.loads(completed.stdout)

    def claim(self) -> dict[str, Any] | None:
        claim: dict[str, Any] | None = self._run(["tts", "claim"])
        return claim

    def complete(self, profile: str, task_id: int, claim_token: str, audio: bytes) -> None:
        self._run(
            [
                "--profile",
                profile,
                "tts",
                "complete",
                "--task",
                str(task_id),
                "--claim",
                claim_token,
            ],
            stdin=audio,
        )

    def fail(self, profile: str, task_id: int, claim_token: str, error: str) -> None:
        self._run(
            ["--profile", profile, "tts", "fail", "--task", str(task_id), "--claim", claim_token]
            + ["--error", error]
        )

    def release(self, profile: str, task_id: int, claim_token: str, reason: str) -> None:
        self._run(
            ["--profile", profile, "tts", "release", "--task", str(task_id), "--claim", claim_token]
            + ["--reason", reason]
        )


class HttpRemote:
    """The owner's ``/api/audio-worker/`` endpoints (see ``audio_worker_api``)."""

    def __init__(self, base_url: str, token: str, *, timeout: float = HTTP_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url
        self.token = token
        self.timeout = timeout

    def _post(self, path: str, *, json_body: Any = None, data: bytes | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self.token}"}
        if data is None:
            data = json.dumps(json_body if json_body is not None else {}).encode()
            headers["Content-Type"] = "application/json"
        else:
            headers["Content-Type"] = "audio/wav"
        request = urllib.request.Request(
            f"{self.base_url}/api/audio-worker/{path}", data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read(500).decode("utf-8", "replace")
            if error.code >= 500:
                raise RemoteUnavailable(f"owner answered HTTP {error.code}") from error
            raise RuntimeError(f"owner answered HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RemoteUnavailable(str(getattr(error, "reason", error))) from error
        return json.loads(body) if body else None

    def claim(self) -> dict[str, Any] | None:
        claim: dict[str, Any] | None = self._post("claim")
        return claim

    def complete(self, profile: str, task_id: int, claim_token: str, audio: bytes) -> None:
        query = urllib.parse.urlencode({"claim_token": claim_token})
        self._post(f"{profile}/{task_id}/complete?{query}", data=audio)

    def fail(self, profile: str, task_id: int, claim_token: str, error: str) -> None:
        self._post(
            f"{profile}/{task_id}/fail", json_body={"claim_token": claim_token, "error": error}
        )

    def release(self, profile: str, task_id: int, claim_token: str, reason: str) -> None:
        self._post(
            f"{profile}/{task_id}/release", json_body={"claim_token": claim_token, "reason": reason}
        )


class Worker:
    def __init__(self, remote: Remote, provider: ProviderProcess) -> None:
        self.remote = remote
        self.provider = provider
        self.stopping = False
        self.synthesizing = False

    def stop(self, _signal: int, _frame: FrameType | None) -> None:
        self.stopping = True
        if self.synthesizing:
            raise ShutdownRequested()

    def step(self) -> bool:
        if self.stopping:
            return False
        claim = self.remote.claim()
        if claim is None:
            return False
        profile, task_id, token = claim["profile_id"], int(claim["task_id"]), claim["claim_token"]
        with tempfile.TemporaryDirectory(prefix="lang-tts-") as directory:
            output = Path(directory) / "audio.tmp.wav"  # the provider requires this suffix
            try:
                request = AudioProviderRequest(**claim["request"], output_path=str(output))
                if self.stopping:
                    raise ShutdownRequested()
                self.synthesizing = True
                try:
                    self.provider.generate(request)
                finally:
                    self.synthesizing = False
                if self.stopping:
                    raise ShutdownRequested()
                validate_wave(output)
            except ShutdownRequested:
                self.provider.close()
                self.remote.release(profile, task_id, token, "audio worker stopping")
                return True
            except Exception as error:
                if self.stopping:
                    self.provider.close()
                    self.remote.release(profile, task_id, token, "audio worker stopping")
                    return True
                message = f"{type(error).__name__}: {error}"
                if any(marker in message for marker in ENVIRONMENT_ERROR_MARKERS):
                    self.remote.release(profile, task_id, token, message)
                    raise RuntimeBroken(message) from error
                self.remote.fail(profile, task_id, token, message)
                print(f"audio {profile}/{task_id} failed: {message}", flush=True)
                return True
            self.remote.complete(profile, task_id, token, output.read_bytes())
        print(f"audio {profile}/{task_id} completed", flush=True)
        return True

    def run(self) -> None:
        while not self.stopping:
            try:
                worked = self.step()
            except RemoteUnavailable as error:
                print(f"owner unreachable, retrying later: {error}", flush=True)
                self._sleep(UNREACHABLE_SECONDS)
                continue
            except RuntimeBroken as error:
                print(f"local audio runtime is broken, retrying later: {error}", flush=True)
                self._sleep(RUNTIME_BROKEN_SECONDS)
                continue
            if not worked:
                self._sleep(IDLE_SECONDS)

    def _sleep(self, seconds: float) -> None:
        # Exiting the provider process is the only complete release of its model and GPU context.
        self.provider.close()
        deadline = time.monotonic() + seconds
        while not self.stopping and time.monotonic() < deadline:
            time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a remote host's lesson audio on this machine"
    )
    tts = settings.load().tts
    parser.add_argument("--host", default=tts.remote_host)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    remote: Remote
    if tts.remote_url:
        if not tts.remote_token:
            raise SystemExit("ARC_LANG_REMOTE_URL needs ARC_LANG_REMOTE_TOKEN (or _FILE).")
        remote = HttpRemote(tts.remote_url, tts.remote_token)
    elif args.host and tts.remote_command:
        remote = SshRemote(args.host, tts.remote_command)
    else:
        raise SystemExit(
            "Set ARC_LANG_REMOTE_URL and ARC_LANG_REMOTE_TOKEN (the owner's audio worker API), or "
            "ARC_LANG_REMOTE_HOST and ARC_LANG_REMOTE_CLI (ssh to the owner's `lang` CLI)."
        )
    command = provider_command()
    if command is None:
        raise SystemExit("Local Qwen runtime is not installed (scripts/install_qwen_tts.sh).")
    provider = ProviderProcess(command, timeout_seconds=SYNTHESIS_TIMEOUT_SECONDS)
    worker = Worker(remote, provider)
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    try:
        if args.once:
            try:
                worker.step()
            except RuntimeBroken as error:
                raise SystemExit(f"local audio runtime is broken: {error}") from error
        else:
            worker.run()
    finally:
        provider.close()


if __name__ == "__main__":
    main()
