from __future__ import annotations

import pytest

from local_rag_backend.core.services.evaluation_models import EvalJudge
from local_rag_backend.infrastructure.evaluation import (
    EvalJudgeRequest,
    EvalJudgeResult,
    run_judge_consensus,
    run_single_judge,
)


class FakeJudgeClient:
    def __init__(self, *, score: float, confidence: float | None = 0.9) -> None:
        self._score = score
        self._confidence = confidence

    def judge(self, *, judge: EvalJudge, request: EvalJudgeRequest) -> EvalJudgeResult:
        return EvalJudgeResult(
            judge=judge,
            score=self._score,
            confidence=self._confidence,
            raw_judgment="ok",
            parsed={"question": request.question},
            latency_ms=10,
        )


class ScoreByJudgeClient:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def judge(self, *, judge: EvalJudge, request: EvalJudgeRequest) -> EvalJudgeResult:
        return EvalJudgeResult(
            judge=judge,
            score=self._scores[judge.judge_id],
            confidence=0.8,
            raw_judgment="ok",
            parsed={"answer": request.answer},
        )


def _judge(judge_id: str = "faithfulness-v1") -> EvalJudge:
    return EvalJudge(
        judge_id=judge_id,
        provider="fake",
        model="fake-model",
        prompt_version="p1",
        rubric_version="r1",
    )


def test_run_single_judge_accepts_valid_client_result() -> None:
    result = run_single_judge(
        judge=_judge(),
        client=FakeJudgeClient(score=0.8),
        request=EvalJudgeRequest(question="q", answer="a"),
    )

    assert result.score == pytest.approx(0.8)
    assert result.parsed == {"question": "q"}


def test_run_single_judge_rejects_score_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="score"):
        run_single_judge(
            judge=_judge(),
            client=FakeJudgeClient(score=1.2),
            request=EvalJudgeRequest(question="q", answer="a"),
        )


def test_run_judge_consensus_supports_weighted_vote_and_variance_abstention() -> None:
    judges = (_judge("a"), _judge("b"))
    request = EvalJudgeRequest(question="q", answer="a")

    consensus = run_judge_consensus(
        judges=judges,
        client=ScoreByJudgeClient({"a": 1.0, "b": 0.0}),
        request=request,
        pass_threshold=0.5,
        weights={"a": 3.0, "b": 1.0},
        variance_threshold=0.18,
    )

    assert consensus.score == pytest.approx(0.75)
    assert consensus.variance == pytest.approx(0.1875)
    assert consensus.abstained is True
    assert consensus.passed is False
