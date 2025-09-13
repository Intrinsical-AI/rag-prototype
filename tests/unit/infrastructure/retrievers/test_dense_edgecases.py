# tests/unit/infrastructure/retrievers/test_dense_edgecases.py
from typing import ClassVar

from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever


class E:
    def embed(self, xs):
        return [[0.0, 0.0]]


class MockIndex:
    id_map: ClassVar = [1]

    def search(self, q, k):
        return [0], [0.0]


class R:
    def get(self, ids):
        return []


def test_dense_k_le_zero_returns_empty():
    retr = DenseFaissRetriever(embedder=E(), faiss_index=MockIndex(), doc_repo=R())
    assert retr.retrieve("q", k=0) == ([], [])

def test_retriever_empty_doc_repo_returns_empty():
    class MockDocRepo:
        def get_documents_by_ids(self, ids):
            return []
    retriever = DenseFaissRetriever(MockIndex(), MockDocRepo())
    docs, scores = retriever.retrieve("q", k=1)
    assert docs == []
    assert scores == []
