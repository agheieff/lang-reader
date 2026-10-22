from __future__ import annotations

import json
from pathlib import Path

from server.generation_usage import task_usage


def _timings(agent: Path, attempt: int, events: list[dict[str, object]], task: int = 7) -> None:
    directory = agent / "jobs" / f"task-{task}-attempt-{attempt}"
    directory.mkdir(parents=True)
    lines = [json.dumps(event) for event in events]
    (directory / "timings.jsonl").write_text("\n".join([*lines, "not json", ""]), encoding="utf-8")


def test_usage_sums_callbacks_across_attempts_and_ignores_host_events(tmp_path: Path) -> None:
    _timings(
        tmp_path,
        1,
        [
            {"kind": "host", "operation": "prepare", "input_tokens": 999},
            {"kind": "callback", "input_tokens": 100, "output_tokens": 10},
            {"kind": "callback", "input_tokens": 50, "cached_input_tokens": 40},
        ],
    )
    _timings(tmp_path, 2, [{"kind": "callback", "input_tokens": 5, "output_tokens": 1}])
    _timings(tmp_path, 1, [{"kind": "callback", "input_tokens": 700}], task=70)

    usage = task_usage(tmp_path, 7)

    assert usage["input_tokens"] == 155 and usage["output_tokens"] == 11
    assert usage["cached_input_tokens"] == 40
    assert usage["calls"] == 3 and usage["calls_reporting_usage"] == 3


def test_unreported_usage_is_distinguishable_from_zero(tmp_path: Path) -> None:
    _timings(tmp_path, 1, [{"kind": "callback", "status": "ok"}, {"kind": "callback"}])
    usage = task_usage(tmp_path, 7)
    assert usage["input_tokens"] == 0
    assert usage["calls"] == 2 and usage["calls_reporting_usage"] == 0


def test_a_task_without_jobs_reports_no_calls(tmp_path: Path) -> None:
    assert task_usage(tmp_path / "missing", 1)["calls"] == 0
