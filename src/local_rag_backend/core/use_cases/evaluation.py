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
    EvalDataset,
    EvalResult,
    run_retrieval_eval as run_retrieval_eval_core,
)
from local_rag_backend.core.services.types import EvalRetrievalConfig


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
