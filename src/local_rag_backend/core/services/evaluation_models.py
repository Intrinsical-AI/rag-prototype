"""Evaluation DTOs and transport-agnostic model types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EvalRetrievalMode = Literal["sparse", "dense", "dual", "hybrid"]


@dataclass(frozen=True)
class EvalDoc:
    external_id: str
    content: str
    source_id: str | None = None


@dataclass(frozen=True)
class EvalQrel:
    external_id: str
    relevance: int = 1


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_external_ids: tuple[str, ...]
    qrels: tuple[EvalQrel, ...] = ()


@dataclass(frozen=True)
class EvalDataset:
    dataset_id: str
    schema_version: int
    docs: tuple[EvalDoc, ...]
    queries: tuple[EvalQuery, ...]


@dataclass(frozen=True)
class EvalRetrievalConfig:
    retrieval_mode: EvalRetrievalMode
    k: int
    candidate_k: int | None = None
    dual_candidate_k: int | None = None
    hybrid_alpha: float | None = None
    reranker_enabled: bool = False


@dataclass(frozen=True)
class EvalBatchSpec:
    name: str
    retrieval_mode: EvalRetrievalMode
    k: int
    candidate_k: int | None = None
    dual_candidate_k: int | None = None
    hybrid_alpha: float | None = None
    reranker_enabled: bool = False
    json_out: str | None = None
    run_out: str | None = None
    report_out: str | None = None
    anomalies_out: str | None = None


@dataclass(frozen=True)
class EvalBatchResult:
    name: str
    result: EvalResult
    json_out: str | None = None
    run_out: str | None = None
    report_out: str | None = None
    anomalies_out: str | None = None


@dataclass(frozen=True)
class EvalRetrievedItem:
    external_id: str
    score: float | None = None


@dataclass(frozen=True)
class EvalRunItem:
    external_id: str
    rank: int
    score: float
    known: bool
    relevance: int = 0
    anomaly: str | None = None


@dataclass(frozen=True)
class EvalQueryMetrics:
    ndcg_at_k: float
    map_at_k: float
    mrr_at_k: float
    precision_at_k: float
    recall_at_k: float


@dataclass(frozen=True)
class EvalAnomaly:
    query_id: str
    kind: str
    external_id: str | None = None
    message: str = ""


@dataclass(frozen=True)
class EvalQueryResult:
    query_id: str
    query_text: str
    qrels: tuple[EvalQrel, ...]
    ranked_docs: tuple[EvalRunItem, ...]
    metrics: EvalQueryMetrics


@dataclass(frozen=True)
class EvalCompareConfig:
    retrieval_mode: EvalRetrievalMode
    candidate_k: int | None = None
    dual_candidate_k: int | None = None
    hybrid_alpha: float | None = None
    reranker_enabled: bool = False


@dataclass(frozen=True)
class EvalCompareThresholds:
    min_delta_ndcg: float = 0.0
    min_delta_map: float = 0.0
    min_delta_mrr: float = 0.0
    max_regression_precision: float = 0.0
    max_regression_recall: float = 0.0


@dataclass(frozen=True)
class EvalCompareSpec:
    baseline: EvalCompareConfig
    candidate: EvalCompareConfig
    thresholds: EvalCompareThresholds = field(default_factory=EvalCompareThresholds)
    k: int = 3
    max_queries: int | None = None


@dataclass(frozen=True)
class EvalResult:
    dataset_id: str
    retrieval_mode: EvalRetrievalMode
    reranker_enabled: bool
    k: int
    queries: int
    ndcg_at_k: float
    map_at_k: float
    mrr_at_k: float
    precision_at_k: float
    recall_at_k: float
    per_query: tuple[EvalQueryResult, ...] = ()
    anomalies: tuple[EvalAnomaly, ...] = ()


@dataclass(frozen=True)
class EvalCompareGate:
    passed: bool
    reasons: tuple[str, ...]
    min_delta_ndcg: float
    min_delta_map: float
    min_delta_mrr: float
    max_regression_precision: float
    max_regression_recall: float


@dataclass(frozen=True)
class EvalCompareDelta:
    ndcg_at_k: float
    map_at_k: float
    mrr_at_k: float
    precision_at_k: float
    recall_at_k: float


@dataclass(frozen=True)
class EvalCompareQueryDelta:
    query_id: str
    baseline: EvalQueryMetrics
    candidate: EvalQueryMetrics
    delta: EvalCompareDelta


@dataclass(frozen=True)
class EvalCompareResult:
    dataset_id: str
    k: int
    baseline: EvalResult
    candidate: EvalResult
    delta: EvalCompareDelta
    gate: EvalCompareGate
    per_query: tuple[EvalCompareQueryDelta, ...] = ()


@dataclass(frozen=True)
class EvalReport:
    run_id: str
    result: EvalResult
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalJudge:
    judge_id: str
    provider: str
    model: str
    prompt_version: str
    rubric_version: str
    temperature: float = 0.0


@dataclass(frozen=True)
class EvalBenchmark:
    benchmark_id: str
    adapter: str
    dataset_id: str
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
