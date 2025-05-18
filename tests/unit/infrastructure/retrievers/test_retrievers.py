import numpy as np
import pytest

from src.core.domain.entities import Document
from src.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from src.infrastructure.retrieval.hybrid import HybridRetriever
from src.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


class DummyEmbedder:
    """Embedder determinista: mapea texto a embedding fijo (hashable para tests)."""

    dim = 2

    def embed(self, texts):
        # “A” siempre [1, 0], “B” siempre [0, 1]
        return [[1, 0] if "A" in t else [0, 1] for t in texts]


class DummyDocRepo:
    def __init__(self):
        self.docs = [Document(id=1, content="Doc A"), Document(id=2, content="Doc B")]

    def get(self, ids):
        return [d for d in self.docs if d.id in ids]

    def get_all_documents(self):
        return self.docs


class DummyFaissIndex:
    def __init__(self):
        self.id_map = [1, 2]
        # Para query=[1,0], devuelve idx=0 (Doc A), query=[0,1], idx=1 (Doc B)

    def search(self, query_vec, k):
        if query_vec == [1, 0]:  # closest: idx 0
            return np.array([0]), np.array([0.0])
        elif query_vec == [0, 1]:  # closest: idx 1
            return np.array([1]), np.array([0.0])
        else:
            return np.array([0]), np.array([999.0])


def test_dense_faiss_retriever_basic():
    retriever = DenseFaissRetriever(
        embedder=DummyEmbedder(),
        faiss_index=DummyFaissIndex(),
        doc_repo=DummyDocRepo(),
    )
    docs, scores = retriever.retrieve("Doc A", k=1)
    assert docs and docs[0].content == "Doc A"
    assert scores[0] == pytest.approx(0.0)
    docs, scores = retriever.retrieve("Doc B", k=1)
    assert docs and docs[0].content == "Doc B"


def test_sparse_bm25_retriever_basic():
    # Sin tokenización real, pero BM25Okapi exige listas de palabras, así que “hackeamos”:

    class DummyBM25:
        def __init__(self):
            pass

        def get_scores(self, query):
            # Devuelve 1.0 si “a” está en query, 0.0 si no.
            return [1.0, 0.0] if "a" in query else [0.0, 1.0]

    class DummySparse(SparseBM25Retriever):
        def __init__(self, documents, doc_ids, doc_repo):
            self.doc_ids = doc_ids
            self.doc_repo = doc_repo
            self.bm25 = DummyBM25()
            self.corpus_is_empty = False

        @staticmethod
        def _tok(text):
            return list(text.lower())

    repo = DummyDocRepo()
    retriever = DummySparse(documents=["Doc A", "Doc B"], doc_ids=[1, 2], doc_repo=repo)
    docs, scores = retriever.retrieve("a", k=1)
    assert len(docs) == 1 and docs[0].content == "Doc A"
    assert scores[0] == 1.0

    docs, scores = retriever.retrieve("b", k=1)
    assert len(docs) == 1 and docs[0].content == "Doc B"
    assert scores[0] == 1.0


def test_hybrid_retriever_merges_and_ranks():
    # Los dense y sparse producen 1 doc cada uno, se fusionan, ambos deben salir en top-2
    class DummyRetriever:
        def __init__(self, docs, scores):
            self._docs, self._scores = docs, scores

        def retrieve(self, query, k=5):
            return self._docs[:k], self._scores[:k]

    doc_a = Document(id=1, content="Doc A")
    doc_b = Document(id=2, content="Doc B")
    dense = DummyRetriever([doc_a], [0.7])
    sparse = DummyRetriever([doc_b], [1.0])
    hybrid = HybridRetriever(dense=dense, sparse=sparse, alpha=0.5)
    docs, scores = hybrid.retrieve("irrelevant", k=2)
    assert set(d.content for d in docs) == {"Doc A", "Doc B"}
    assert len(scores) == 2
