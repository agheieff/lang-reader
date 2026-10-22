"""Model token usage of a generation task, summed from its private timing records."""

from __future__ import annotations

import json
from pathlib import Path

from server.agent_command import USAGE_FIELDS


def task_usage(agent_directory: Path, task_id: int, attempt: int | None = None) -> dict[str, int]:
    """Sum the callback usage of every attempt of one task, or of just one attempt.

    ``calls`` counts model calls and ``calls_reporting_usage`` those whose command reported
    token counts, so a total of zero is distinguishable from "nothing was reported".
    """

    totals = dict.fromkeys(USAGE_FIELDS, 0)
    calls = reporting = 0
    for timings in sorted(
        (agent_directory / "jobs").glob(
            f"task-{task_id}-attempt-{'*' if attempt is None else attempt}/timings.jsonl"
        )
    ):
        for line in timings.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("kind") != "callback":
                continue
            calls += 1
            found = [
                (field, value)
                for field in USAGE_FIELDS
                if type(value := event.get(field)) is int and value >= 0
            ]
            reporting += bool(found)
            for field, value in found:
                totals[field] += value
    return {**totals, "calls": calls, "calls_reporting_usage": reporting}
