"""Optional evaluation adapters."""

from local_rag_backend.infrastructure.evaluation.benchmarks import (
    eval_dataset_from_ir_mappings,
)
from local_rag_backend.infrastructure.evaluation.llm_judge import (
    EvalConsensusResult,
    EvalJudgeClient,
    EvalJudgeRequest,
    EvalJudgeResult,
    run_judge_consensus,
    run_single_judge,
)
from local_rag_backend.infrastructure.evaluation.ragas_adapter import (
    EvalAdapterUnavailableError,
    RagasEvaluationAdapter,
)

__all__ = [
    "EvalAdapterUnavailableError",
    "EvalConsensusResult",
    "EvalJudgeClient",
    "EvalJudgeRequest",
    "EvalJudgeResult",
    "RagasEvaluationAdapter",
    "eval_dataset_from_ir_mappings",
    "run_judge_consensus",
    "run_single_judge",
]
