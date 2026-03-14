# tests/unit/app/services/test_evaluation.py

from local_rag_backend.composition.adapters import (
    build_eval_retriever_factory_port,
    build_eval_storage_port,
)
from local_rag_backend.core.services.evaluation import load_eval_dataset
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
    assert 0.0 <= res.hit_rate <= 1.0
    assert 0.0 <= res.mrr <= 1.0
