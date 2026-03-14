# tests/unit/app/services/test_evaluation.py

from local_rag_backend.composition.adapters import (
    build_eval_retriever_factory_port,
    build_eval_storage_port,
)
from local_rag_backend.core.ports import EvalDatasetDocInput
from local_rag_backend.core.services.evaluation import load_eval_dataset, run_retrieval_eval as run_core_eval
from local_rag_backend.core.use_cases.evaluation import run_retrieval_eval


def test_run_retrieval_eval_app_service_passes_on_default_repo_dataset() -> None:
    ds = load_eval_dataset()
    res = run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(),
        eval_retriever_factory_port=build_eval_retriever_factory_port(),
        retrieval_mode="sparse",
        reranker_enabled=True,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
    )
    assert res.queries > 0
    assert 0.0 <= res.ndcg_at_k <= 1.0
    assert 0.0 <= res.map_at_k <= 1.0
    assert 0.0 <= res.mrr_at_k <= 1.0
    assert 0.0 <= res.precision_at_k <= 1.0
    assert 0.0 <= res.recall_at_k <= 1.0


def test_run_retrieval_eval_app_service_matches_core_semantics() -> None:
    ds = load_eval_dataset()
    storage = build_eval_storage_port()
    storage.upsert_dataset_docs(
        dataset_id=ds.dataset_id,
        docs=tuple(
            EvalDatasetDocInput(
                external_id=d.external_id,
                content=d.content,
                source_id=d.source_id,
                metadata={"dataset_id": ds.dataset_id},
            )
            for d in ds.docs
        ),
    )
    retriever = build_eval_retriever_factory_port().build_sparse_retriever(
        storage=storage,
        reranker_enabled=True,
        candidate_k=20,
        strategy="overlap_v1",
    )

    def _retrieve_external_ids(query: str, top_k: int) -> list[str]:
        docs, _scores = retriever.retrieve(query, top_k)
        return [
            str(external_id)
            for external_id in (getattr(d, "external_id", None) for d in docs)
            if external_id is not None and str(external_id).strip()
        ]

    core_res = run_core_eval(
        dataset=ds,
        retrieve_external_ids=_retrieve_external_ids,
        retrieval_mode="sparse",
        k=3,
        reranker_enabled=True,
    )
    app_res = run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(),
        eval_retriever_factory_port=build_eval_retriever_factory_port(),
        retrieval_mode="sparse",
        k=3,
        reranker_enabled=True,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
    )

    assert app_res == core_res
