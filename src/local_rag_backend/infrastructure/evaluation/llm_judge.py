"""LLM-as-a-judge adapter contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from local_rag_backend.core.services.evaluation_models import EvalJudge


@dataclass(frozen=True)
class EvalJudgeRequest:
    question: str
    answer: str
    contexts: tuple[str, ...] = ()
    reference_answer: str | None = None


@dataclass(frozen=True)
class EvalJudgeResult:
    judge: EvalJudge
    score: float
    confidence: float | None
    raw_judgment: str
    parsed: dict[str, object]
    cost_usd: float | None = None
    latency_ms: int | None = None


@dataclass(frozen=True)
class EvalConsensusResult:
    results: tuple[EvalJudgeResult, ...]
    score: float
    passed: bool
    variance: float
    abstained: bool


class EvalJudgeClient(Protocol):
    def judge(self, *, judge: EvalJudge, request: EvalJudgeRequest) -> EvalJudgeResult: ...


def run_single_judge(
    *,
    judge: EvalJudge,
    client: EvalJudgeClient,
    request: EvalJudgeRequest,
) -> EvalJudgeResult:
    result = client.judge(judge=judge, request=request)
    if result.judge != judge:
        raise ValueError("Judge client returned a result for a different judge.")
    if not 0.0 <= float(result.score) <= 1.0:
        raise ValueError("Judge score must be between 0.0 and 1.0.")
    if result.confidence is not None and not 0.0 <= float(result.confidence) <= 1.0:
        raise ValueError("Judge confidence must be between 0.0 and 1.0.")
    return result


def run_judge_consensus(
    *,
    judges: Sequence[EvalJudge],
    client: EvalJudgeClient,
    request: EvalJudgeRequest,
    pass_threshold: float = 0.5,
    weights: Mapping[str, float] | None = None,
    variance_threshold: float | None = None,
) -> EvalConsensusResult:
    """Run a weighted score consensus; variance abstention uses the same weights."""
    if not judges:
        raise ValueError("At least one judge is required for consensus.")
    if not 0.0 <= float(pass_threshold) <= 1.0:
        raise ValueError("pass_threshold must be between 0.0 and 1.0.")
    results = tuple(
        run_single_judge(judge=judge, client=client, request=request) for judge in judges
    )
    score_by_judge = {result.judge.judge_id: float(result.score) for result in results}
    weight_by_judge = {judge_id: float(weight) for judge_id, weight in dict(weights or {}).items()}
    total_weight = 0.0
    weighted_score = 0.0
    for judge_id, score in score_by_judge.items():
        weight = weight_by_judge.get(judge_id, 1.0)
        if weight < 0.0:
            raise ValueError("Consensus weights must be non-negative.")
        total_weight += weight
        weighted_score += score * weight
    if total_weight <= 0.0:
        raise ValueError("At least one consensus weight must be positive.")
    consensus_score = weighted_score / total_weight
    variance = (
        sum(
            weight_by_judge.get(judge_id, 1.0) * (score - consensus_score) ** 2
            for judge_id, score in score_by_judge.items()
        )
        / total_weight
    )
    abstained = variance_threshold is not None and variance > float(variance_threshold)
    return EvalConsensusResult(
        results=results,
        score=consensus_score,
        passed=bool(consensus_score >= float(pass_threshold) and not abstained),
        variance=variance,
        abstained=abstained,
    )
