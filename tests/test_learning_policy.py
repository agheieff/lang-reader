from __future__ import annotations

from typing import Any

import pytest

from server.learning_policy import TermBandPolicy


@pytest.mark.parametrize(
    "policy",
    [
        {"expected_min_mastery": 0.7},
        {"mastered_min_mastery": 0.5},
        {"expected_min_exposures": -1},
        {"base_expected_frequency_rank": 1_000, "max_expected_frequency_rank": 500},
        {"uncertain_frequency_multiplier": 0.5},
    ],
)
def test_term_band_policy_rejects_invalid_configuration(policy: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TermBandPolicy(**policy)
