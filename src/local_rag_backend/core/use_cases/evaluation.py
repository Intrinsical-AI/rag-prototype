"""Application-layer orchestration for offline retrieval evaluation."""

from __future__ import annotations

from typing import Any

from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    retrieval_result_from_pairs,
)
from local_rag_backend.core.ports import (
    EvalDatasetDocInput,
    EvalRetrieverFactoryPort,
    EvalStoragePort,
)
from local_rag_backend.core.services.evaluation import (
    EvalCompareResult,
    EvalDataset,
    EvalResult,
    compare_eval_results,
    run_retrieval_eval as run_retrieval_eval_core,
)
from local_rag_backend.core.services.types import EvalCompareConfig, EvalRetrievalConfig


def _coerce_retrieval_result(
    raw_result: Any,
    *,
    request: RetrievalRequest,
) -> RetrievalResult:
    if isinstance(raw_result, RetrievalResult):
        return raw_result
    if (
        isinstance(raw_result, tuple)
        and len(raw_result) == 2
        and isinstance(raw_result[0], (list, tuple))
        and isinstance(raw_result[1], (list, tuple))
    ):
        docs, scores = raw_result
        return retrieval_result_from_pairs(
            docs=docs,
            scores=scores,
            mode_used=request.mode,
            backend_used="legacy_eval",
        )
    raise RuntimeError(f"Unsupported eval retriever response type: {type(raw_result)!r}")


def _validate_eval_options(
    *,
    retrieval_mode: str,
    candidate_k: int | None,
    dual_candidate_k: int | None,
    hybrid_alpha: float | None,
) -> None:
    if candidate_k is not None and retrieval_mode != "dense":
        raise ValueError("--candidate-k is supported only with retrieval_mode=dense")
    if dual_candidate_k is not None and retrieval_mode != "dual":
        raise ValueError("--dual-candidate-k is supported only with retrieval_mode=dual")
    if hybrid_alpha is not None and retrieval_mode != "hybrid":
        raise ValueError("--hybrid-alpha is supported only with retrieval_mode=hybrid")
    if candidate_k is not None and int(candidate_k) <= 0:
        raise ValueError("candidate_k must be positive")
    if dual_candidate_k is not None and int(dual_candidate_k) <= 0:
        raise ValueError("dual_candidate_k must be positive")
    if hybrid_alpha is not None and not 0.0 <= float(hybrid_alpha) <= 1.0:
        raise ValueError("hybrid_alpha must be between 0.0 and 1.0")


def run_retrieval_eval(
    *,
    dataset: EvalDataset,
    eval_storage_port: EvalStoragePort,
    eval_retriever_factory_port: EvalRetrieverFactoryPort,
    retrieval_mode: str = "sparse",
    k: int = 3,
    candidate_k: int | None = None,
    reranker_enabled: bool = False,
    dual_candidate_k: int | None = None,
    hybrid_alpha: float | None = None,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
) -> EvalResult:
    if retrieval_mode not in {"sparse", "dense", "dual", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
    if k <= 0:
        raise ValueError("k must be positive")
    _validate_eval_options(
        retrieval_mode=str(retrieval_mode),
        candidate_k=candidate_k,
        dual_candidate_k=dual_candidate_k,
        hybrid_alpha=hybrid_alpha,
    )

    eval_storage_port.upsert_dataset_docs(
        dataset_id=dataset.dataset_id,
        docs=tuple(
            EvalDatasetDocInput(
                external_id=d.external_id,
                content=d.content,
                source_id=d.source_id,
                metadata={"dataset_id": dataset.dataset_id},
            )
            for d in dataset.docs
        ),
    )

    retriever = eval_retriever_factory_port.build_retriever(
        storage=eval_storage_port,
        config=EvalRetrievalConfig(
            retrieval_mode=str(retrieval_mode),
            k=int(k),
            candidate_k=(int(candidate_k) if candidate_k is not None else None),
            dual_candidate_k=(int(dual_candidate_k) if dual_candidate_k is not None else None),
            hybrid_alpha=(float(hybrid_alpha) if hybrid_alpha is not None else None),
            reranker_enabled=bool(reranker_enabled),
        ),
        reranker_candidate_k=reranker_candidate_k,
        reranker_strategy=reranker_strategy,
    )

    def _retrieve_external_ids(query: str, top_k: int) -> list[str]:
        request = RetrievalRequest(
            query=query,
            top_k=top_k,
            mode=str(retrieval_mode),  # type: ignore[arg-type]
            candidate_k=(int(candidate_k) if candidate_k is not None else None),
            dual_candidate_k=(int(dual_candidate_k) if dual_candidate_k is not None else None),
        )
        retrieval = _coerce_retrieval_result(retriever.retrieve(request), request=request)
        return [
            str(external_id)
            for external_id in (
                getattr(document, "external_id", None) for document in retrieval.documents
            )
            if external_id is not None and str(external_id).strip()
        ]

    return run_retrieval_eval_core(
        dataset=dataset,
        retrieve_external_ids=_retrieve_external_ids,
        retrieval_mode=retrieval_mode,
        k=k,
        reranker_enabled=reranker_enabled,
        max_queries=max_queries,
    )


def compare_retrieval_eval(
    *,
    dataset: EvalDataset,
    eval_storage_port: EvalStoragePort,
    eval_retriever_factory_port: EvalRetrieverFactoryPort,
    baseline: EvalCompareConfig,
    candidate: EvalCompareConfig,
    k: int = 3,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
    min_delta_ndcg: float = 0.0,
    min_delta_map: float = 0.0,
    min_delta_mrr: float = 0.0,
    max_regression_precision: float = 0.0,
    max_regression_recall: float = 0.0,
) -> EvalCompareResult:
    baseline_result = run_retrieval_eval(
        dataset=dataset,
        eval_storage_port=eval_storage_port,
        eval_retriever_factory_port=eval_retriever_factory_port,
        retrieval_mode=baseline.retrieval_mode,
        k=k,
        candidate_k=baseline.candidate_k,
        reranker_enabled=baseline.reranker_enabled,
        dual_candidate_k=baseline.dual_candidate_k,
        hybrid_alpha=baseline.hybrid_alpha,
        reranker_candidate_k=reranker_candidate_k,
        reranker_strategy=reranker_strategy,
        max_queries=max_queries,
    )
    candidate_result = run_retrieval_eval(
        dataset=dataset,
        eval_storage_port=eval_storage_port,
        eval_retriever_factory_port=eval_retriever_factory_port,
        retrieval_mode=candidate.retrieval_mode,
        k=k,
        candidate_k=candidate.candidate_k,
        reranker_enabled=candidate.reranker_enabled,
        dual_candidate_k=candidate.dual_candidate_k,
        hybrid_alpha=candidate.hybrid_alpha,
        reranker_candidate_k=reranker_candidate_k,
        reranker_strategy=reranker_strategy,
        max_queries=max_queries,
    )
    return compare_eval_results(
        baseline=baseline_result,
        candidate=candidate_result,
        min_delta_ndcg=min_delta_ndcg,
        min_delta_map=min_delta_map,
        min_delta_mrr=min_delta_mrr,
        max_regression_precision=max_regression_precision,
        max_regression_recall=max_regression_recall,
    )
