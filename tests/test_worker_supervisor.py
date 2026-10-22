from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

from watchfiles import Change

from server.worker_supervisor import WorkerSupervisor, restart_delay, worker_source_filter


class FakeProcess:
    next_pid = 1_000

    def __init__(self, polls: list[int | None]) -> None:
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.polls = polls
        self.returncode: int | None = None
        self.signals: list[int] = []
        self.wait_count = 0
        self.kill_count = 0

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self.polls:
            result = self.polls.pop(0)
            if result is not None:
                self.returncode = result
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_count += 1
        if self.returncode is None:
            self.returncode = -signal.SIGTERM
        return self.returncode

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -signal.SIGKILL


def test_worker_source_filter_is_narrow(tmp_path: Path) -> None:
    server = tmp_path / "server"
    cases = {
        server / "agent_worker.py": True,
        server / "nested" / "module.py": True,
        tmp_path / "packs" / "zh" / "rules" / "zh.toml": True,
        tmp_path / "packs" / "de" / "grammar" / "de.toml": True,
        server / "other.toml": False,
        server / "static" / "app.js": False,
        server / "static" / "generated.py": False,
        tmp_path / "tests" / "test_generation.py": False,
    }

    for path, expected in cases.items():
        assert worker_source_filter(Change.modified, str(path), server_dir=server) is expected


def test_restart_delay_is_exponential_and_bounded() -> None:
    assert [restart_delay(index) for index in range(1, 8)] == [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        30.0,
    ]
    assert restart_delay(50) == 30.0


def test_unexpected_exits_restart_with_backoff_and_success_resets_it() -> None:
    processes = [FakeProcess([1]), FakeProcess([1]), FakeProcess([0]), FakeProcess([1])]
    delays: list[float] = []

    def spawn() -> Any:
        return processes.pop(0)

    def wait_for_change(delay: float) -> bool:
        delays.append(delay)
        return False

    supervisor = WorkerSupervisor(
        spawn_child=spawn,
        poll_source_changes=lambda: False,
        wait_for_source_change=wait_for_change,
        idle_poll_seconds=2,
    )

    assert supervisor.run(max_cycles=4) == 0
    assert delays == [0.5, 1.0, 2, 0.5]
    assert not processes


def test_source_change_waits_for_active_child_then_starts_fresh_worker() -> None:
    first = FakeProcess([None, None, 0])
    second = FakeProcess([0])
    processes = [first, second]
    changes = iter((True, False))
    delays: list[float] = []

    supervisor = WorkerSupervisor(
        spawn_child=lambda: processes.pop(0),  # type: ignore[arg-type]
        poll_source_changes=lambda: next(changes),
        wait_for_source_change=lambda delay: delays.append(delay) or False,
        idle_poll_seconds=2,
    )

    assert supervisor.run(max_cycles=2) == 0
    assert first.signals == []
    assert first.wait_count == 1
    assert second.wait_count == 1
    assert delays == [2]
    assert not processes


def test_stop_signal_is_forwarded_and_child_is_reaped() -> None:
    child = FakeProcess([None])
    supervisor: WorkerSupervisor

    def poll_changes() -> bool:
        supervisor.request_stop(signal.SIGTERM)
        return False

    supervisor = WorkerSupervisor(
        spawn_child=lambda: child,  # type: ignore[arg-type]
        poll_source_changes=poll_changes,
        wait_for_source_change=lambda _delay: False,
    )

    assert supervisor.run(max_cycles=2) == 0
    assert signal.SIGTERM in child.signals
    assert child.wait_count == 1
    assert child.kill_count == 0
