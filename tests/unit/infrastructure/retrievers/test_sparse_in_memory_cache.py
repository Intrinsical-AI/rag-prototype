from __future__ import annotations

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


class CountingRepo:
    def __init__(self) -> None:
        self.docs = [Document(id=1, content="alpha one"), Document(id=2, content="beta two")]
        self.get_calls = 0

    def get(self, ids):
        self.get_calls += 1
        s = set(ids)
        return [d for d in self.docs if d.id in s]


def test_sparse_retriever_uses_in_memory_doc_cache_after_initial_load():
    repo = CountingRepo()
    retriever = SparseBM25Retriever(
        documents=["alpha one", "beta two"],
        doc_ids=[1, 2],
        doc_repo=repo,
    )

    # One call during retriever construction to populate in-memory cache.
    assert repo.get_calls == 1

    docs1, _scores1 = retriever.retrieve("alpha", k=1)
    docs2, _scores2 = retriever.retrieve("beta", k=1)

    assert docs1 and docs1[0].id == 1
    assert docs2 and docs2[0].id == 2
    # No per-query DB roundtrip once cache is populated.
    assert repo.get_calls == 1
