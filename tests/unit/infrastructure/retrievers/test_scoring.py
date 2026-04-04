from __future__ import annotations

import math

import pytest

from local_rag_backend.infrastructure.retrieval.scoring import (
    normalize_min_max_scores,
    validate_normalized_scores,
)


def test_normalize_min_max_scores_handles_empty_singleton_and_flat_inputs() -> None:
    assert normalize_min_max_scores([]) == []
    assert normalize_min_max_scores([42.0]) == [1.0]
    assert normalize_min_max_scores([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_normalize_min_max_scores_scales_non_flat_values() -> None:
    assert normalize_min_max_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_validate_normalized_scores_rejects_invalid_values(bad_score: float) -> None:
    with pytest.raises(ValueError):
        validate_normalized_scores([0.5, bad_score], source="test")


def test_validate_normalized_scores_accepts_valid_values() -> None:
    validate_normalized_scores([0.0, 0.25, 1.0], source="test")
