from __future__ import annotations

import rank_bm25

from local_rag_backend.composition.adapters import build_retriever_from_settings
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.settings import Settings


def test_sparse_bm25_is_built_once_for_unfiltered_service_queries(monkeypatch) -> None:
    builds = 0
    real_bm25 = rank_bm25.BM25Okapi

    class _CountingBM25(real_bm25):
        def __init__(self, corpus):
            nonlocal builds
            builds += 1
            super().__init__(corpus)

    monkeypatch.setattr(rank_bm25, "BM25Okapi", _CountingBM25)
    docs = [
        Document(DocId("one"), "capital france paris", metadata={"scope": "public"}),
        Document(DocId("two"), "python auth guard", metadata={"scope": "public"}),
    ]

    class _Repo:
        def get_all_documents(self):
            return docs

        def get(self, ids):
            return [doc for doc in docs if doc.id in ids]

    retriever = build_retriever_from_settings(
        settings_obj=Settings(),
        retrieval_mode="sparse",
        doc_repo=_Repo(),
        dense_embedder_factory=lambda: None,
    )
    retriever.retrieve(RetrievalRequest(query="france", top_k=1))
    retriever.retrieve(RetrievalRequest(query="python", top_k=1))

    assert builds == 1

    retriever.retrieve(
        RetrievalRequest(
            query="france",
            top_k=1,
            filters=(RetrievalFilter("scope", ("public",)),),
        )
    )
    assert builds == 2  # Filtered queries still build a filtered BM25 index.
