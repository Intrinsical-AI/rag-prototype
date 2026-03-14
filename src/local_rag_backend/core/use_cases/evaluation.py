"""Application-layer orchestration for offline retrieval evaluation."""

from __future__ import annotations

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


def run_retrieval_eval(
    *,
    dataset: EvalDataset,
    eval_storage_port: EvalStoragePort,
    eval_retriever_factory_port: EvalRetrieverFactoryPort,
    retrieval_mode: str = "sparse",
    k: int = 3,
    reranker_enabled: bool = False,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
) -> EvalResult:
    if retrieval_mode != "sparse":
        raise ValueError(
            "This offline IR evaluation currently supports retrieval_mode=sparse only."
        )
    if k <= 0:
        raise ValueError("k must be positive")

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

    retriever = eval_retriever_factory_port.build_sparse_retriever(
        storage=eval_storage_port,
        reranker_enabled=reranker_enabled,
        candidate_k=reranker_candidate_k,
        strategy=reranker_strategy,
    )

    def _retrieve_external_ids(query: str, top_k: int) -> list[str]:
        docs, _scores = retriever.retrieve(query, top_k)
        return [
            str(external_id)
            for external_id in (getattr(d, "external_id", None) for d in docs)
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
