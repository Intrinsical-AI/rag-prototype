"""Application-layer orchestration for offline retrieval evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from local_rag_backend.core.services.evaluation import (
    EvalDataset,
    EvalResult,
    run_retrieval_eval as run_retrieval_eval_core,
)
from local_rag_backend.core.services.reranking import RerankingRetriever
from local_rag_backend.infrastructure.persistence.sql import base as db_base
from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever

if TYPE_CHECKING:
    from local_rag_backend.core.ports import RetrieverPort


def _build_ephemeral_doc_repo() -> SqlDocumentStorage:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    from local_rag_backend.infrastructure.persistence.sql import (  # noqa: F401
        models as _models,
    )

    db_base.ensure_sqlite_schema_compatible(engine_to_use=engine)
    return SqlDocumentStorage(session_factory=session_local)


def _build_sparse_retriever(
    *, doc_repo: SqlDocumentStorage, reranker_enabled: bool, candidate_k: int, strategy: str
) -> RetrieverPort:
    docs = doc_repo.get_all_documents()
    corpus = [d.content for d in docs]
    doc_ids = [d.id for d in docs]
    base: RetrieverPort = SparseBM25Retriever(
        documents=corpus,
        doc_ids=doc_ids,
        doc_repo=doc_repo,
        preloaded_docs=docs,
    )
    if reranker_enabled:
        return RerankingRetriever(base, candidate_k=candidate_k, strategy=strategy)
    return base


def run_retrieval_eval(
    *,
    dataset: EvalDataset,
    retrieval_mode: str = "sparse",
    k: int = 3,
    reranker_enabled: bool = False,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
) -> EvalResult:
    if retrieval_mode != "sparse":
        raise ValueError(
            "This eval currently supports retrieval_mode=sparse only (dependency-free)."
        )
    if k <= 0:
        raise ValueError("k must be positive")

    doc_repo = _build_ephemeral_doc_repo()
    items = [
        SqlDocumentStorage.UpsertDoc(
            external_id=d.external_id,
            content=d.content,
            source_id=d.source_id,
            metadata={"dataset_id": dataset.dataset_id},
        )
        for d in dataset.docs
    ]
    results, _changed, _updated = doc_repo.upsert_documents_by_external_id(items)
    known_external_ids = {r.external_id for r in results}

    retriever = _build_sparse_retriever(
        doc_repo=doc_repo,
        reranker_enabled=reranker_enabled,
        candidate_k=reranker_candidate_k,
        strategy=reranker_strategy,
    )

    def _retrieve_external_ids(query: str, top_k: int) -> list[str]:
        docs, _scores = retriever.retrieve(query, top_k)
        return [
            str(d.external_id)
            for d in docs
            if getattr(d, "external_id", None) and str(d.external_id) in known_external_ids
        ]

    return run_retrieval_eval_core(
        dataset=dataset,
        retrieve_external_ids=_retrieve_external_ids,
        retrieval_mode=retrieval_mode,
        k=k,
        reranker_enabled=reranker_enabled,
        max_queries=max_queries,
    )
