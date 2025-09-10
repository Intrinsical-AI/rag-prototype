# tests/unit/infrastructure/retrievers/test_dense_edgecases.py
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever

class E:
    def embed(self, xs): return [[0.0,0.0]]

class I:
    id_map=[1]
    def search(self, q, k): return [0], [0.0]

class R:
    def get(self, ids): return []

def test_dense_k_le_zero_returns_empty():
    retr = DenseFaissRetriever(embedder=E(), faiss_index=I(), doc_repo=R())
    assert retr.retrieve("q", k=0) == ([], [])
