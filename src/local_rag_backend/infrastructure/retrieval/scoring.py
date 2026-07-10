"""Shared score normalization and validation helpers for retrieval backends."""

from __future__ import annotations

import math
from collections.abc import Sequence

SCORE_NORMALIZATION_EPSILON = 1e-6


def normalize_min_max_scores(
    scores: Sequence[float],
    *,
    flat_value: float = 0.0,
    singleton_value: float = 1.0,
) -> list[float]:
    values = [float(score) for score in scores]
    if not values:
        return []
    if len(values) == 1:
        return [float(singleton_value)]
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return [float(flat_value)] * len(values)
    return [(score - min_score) / (max_score - min_score) for score in values]


def validate_normalized_scores(
    scores: Sequence[float],
    *,
    source: str,
    epsilon: float = SCORE_NORMALIZATION_EPSILON,
) -> None:
    for index, raw_score in enumerate(scores):
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(
                f"{source} returned non-finite score at position {index}: {raw_score!r}. "
                "Hybrid inputs must be finite and normalized to [0, 1]."
            )
        if score < -float(epsilon) or score > 1.0 + float(epsilon):
            raise ValueError(
                f"{source} returned score {raw_score!r} at position {index} outside [0, 1]. "
                "Hybrid inputs must be finite and normalized to [0, 1]."
            )
