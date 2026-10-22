from __future__ import annotations

from typing import Any

import pytest

from server.generation_lexical_stage import LexicalExecutionPolicy


@pytest.mark.parametrize(
    "policy",
    [
        {"parallel_callbacks": 0},
        {"local_repair_limit": -1},
        {"conflict_chunk_size": 0},
        {"batch_size": 0},
        {"batch_size": 17},
    ],
)
def test_lexical_execution_policy_rejects_invalid_bounds(policy: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LexicalExecutionPolicy(**policy)
