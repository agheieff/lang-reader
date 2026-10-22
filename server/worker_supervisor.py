"""Restartable development supervisor for the durable generation worker."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path
from types import FrameType

from watchfiles import Change, watch

from server import settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = Path(__file__).resolve().parent
WATCH_TIMEOUT_MS = 250
IDLE_POLL_SECONDS = 2.0
INITIAL_RESTART_SECONDS = 0.5
MAX_RESTART_SECONDS = 30.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0

SpawnChild = Callable[[], subprocess.Popen[bytes]]
PollSourceChanges = Callable[[], bool]
WaitForSourceChange = Callable[[float], bool]


def worker_source_filter(
    _change: Change,
    path: str,
    *,
    server_dir: Path = SERVER_DIR,
) -> bool:
    """Watch worker Python and declarative language files, but never built frontend assets."""

    root = server_dir.resolve()
    candidate = Path(path).resolve()
    # Language packs hold the declarative rules the worker reads; they live beside the package.
    if candidate.suffix == ".toml":
        return candidate.is_relative_to(root.parent / "packs")
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    if relative.parts and relative.parts[0] == "static":
        return False
    return candidate.suffix == ".py"


def restart_delay(consecutive_failures: int) -> float:
    if consecutive_failures < 1:
        raise ValueError("consecutive_failures must be positive")
    exponent = min(consecutive_failures - 1, 10)
    exponential_delay = INITIAL_RESTART_SECONDS * float(2**exponent)
    return min(MAX_RESTART_SECONDS, exponential_delay)


def _idle_poll_seconds() -> float:
    return float(settings.load().agent.poll_seconds)


def _spawn_worker() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "server.agent_worker", "--once"],
        cwd=PROJECT_ROOT,
    )


class WorkerSupervisor:
    """Run one fresh worker process per task without interrupting an active callback."""

    def __init__(
        self,
        *,
        spawn_child: SpawnChild = _spawn_worker,
        poll_source_changes: PollSourceChanges | None = None,
        wait_for_source_change: WaitForSourceChange | None = None,
        idle_poll_seconds: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.stop_event = stop_event or threading.Event()
        self.spawn_child = spawn_child
        self.idle_poll_seconds = (
            _idle_poll_seconds() if idle_poll_seconds is None else idle_poll_seconds
        )
        if self.idle_poll_seconds <= 0:
            raise ValueError("idle_poll_seconds must be positive")
        self._watcher: Iterator[set[tuple[Change, str]]] | None = None
        self.poll_source_changes = poll_source_changes or self._poll_source_changes
        self.wait_for_source_change = wait_for_source_change or self._wait_for_source_change
        self.child: subprocess.Popen[bytes] | None = None

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    def request_stop(self, signum: int, _frame: FrameType | None = None) -> None:
        self.stop_event.set()
        child = self.child
        if child is not None and child.poll() is None:
            with suppress(ProcessLookupError):
                child.send_signal(signum)

    def run(self, *, max_cycles: int | None = None) -> int:
        consecutive_failures = 0
        cycles = 0
        while not self.stop_event.is_set() and (max_cycles is None or cycles < max_cycles):
            source_changed = False
            try:
                child = self.spawn_child()
            except OSError as error:
                exit_code = 1
                print(f"Could not start generation worker: {error}", file=sys.stderr, flush=True)
            else:
                self.child = child
                exit_code, source_changed = self._monitor_child(child)
                self.child = None

            if self.stop_event.is_set():
                break
            if source_changed:
                consecutive_failures = 0
                delay = 0.0
                print(
                    "Generation worker source changed; reloading after the current task.",
                    file=sys.stderr,
                    flush=True,
                )
            elif exit_code == 0:
                consecutive_failures = 0
                delay = self.idle_poll_seconds
            else:
                consecutive_failures += 1
                delay = restart_delay(consecutive_failures)
                print(
                    f"Generation worker exited with status {exit_code}; restarting in {delay:g}s.",
                    file=sys.stderr,
                    flush=True,
                )

            cycles += 1
            if delay > 0 and self.wait_for_source_change(delay):
                consecutive_failures = 0
        return 0

    def _monitor_child(self, child: subprocess.Popen[bytes]) -> tuple[int, bool]:
        source_changed = False
        while not self.stop_event.is_set():
            exit_code = child.poll()
            if exit_code is not None:
                return child.wait(), source_changed
            source_changed = self.poll_source_changes() or source_changed

        if child.poll() is None:
            with suppress(ProcessLookupError):
                child.send_signal(signal.SIGTERM)
        try:
            return child.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS), source_changed
        except subprocess.TimeoutExpired:
            child.kill()
            return child.wait(), source_changed

    def _poll_source_changes(self) -> bool:
        if self._watcher is None:
            self._watcher = watch(
                SERVER_DIR,
                watch_filter=worker_source_filter,
                stop_event=self.stop_event,
                debounce=200,
                step=50,
                rust_timeout=WATCH_TIMEOUT_MS,
                yield_on_timeout=True,
                raise_interrupt=False,
            )
        try:
            return bool(next(self._watcher))
        except StopIteration:
            return False

    def _wait_for_source_change(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            if self.poll_source_changes():
                return True
        return False


def main() -> None:
    try:
        supervisor = WorkerSupervisor()
        supervisor.install_signal_handlers()
        raise SystemExit(supervisor.run())
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
