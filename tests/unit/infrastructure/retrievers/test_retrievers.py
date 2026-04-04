import numpy as np
import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalRequest, RetrievalResult, RetrievedDoc
from local_rag_backend.infrastructure.retrieval.dense_vector import DenseVectorRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


class DummyEmbedder:
    """Embedder determinista: mapea texto a embedding fijo (hashable para tests)."""

    dim = 2

    def embed(self, texts):
        # "A" siempre [1, 0], "B" siempre [0, 1]
        return [[1, 0] if "A" in t else [0, 1] for t in texts]


class DummyDocRepo:
    def __init__(self):
        self.docs = [Document(id=1, content="Doc A"), Document(id=2, content="Doc B")]

    def get(self, ids):
        return [d for d in self.docs if d.id in ids]

    def get_all_documents(self):
        return self.docs


class DummyVectorIndex:
    def __init__(self):
        self.id_map = [1, 2]
        # To query=[1,0], returns idx=0 (Doc A), to query=[0,1], returns idx=1 (Doc B)

    def search(self, query_vec, k):
        if query_vec == [1, 0]:  # closest: idx 0
            return np.array([0]), np.array([0.0])
        elif query_vec == [0, 1]:  # closest: idx 1
            return np.array([1]), np.array([0.0])
        else:
            return np.array([0]), np.array([999.0])

    def similar(self, vector, k):
        # Return (doc_id, similarity_score) pairs
        if vector == [1, 0]:  # closest: Doc A (id=1)
            return [(1, 1.0)]
        elif vector == [0, 1]:  # closest: Doc B (id=2)
            return [(2, 1.0)]
        else:
            return [(1, 0.5)]


def test_dense_vector_retriever_basic():
    retriever = DenseVectorRetriever(
        embedder=DummyEmbedder(),
        vector_repo=DummyVectorIndex(),
        doc_repo=DummyDocRepo(),
    )
    result = retriever.retrieve(RetrievalRequest(query="Doc A", top_k=1, mode="dense"))
    docs, scores = result.documents, result.scores
    assert docs and docs[0].content == "Doc A"
    # With similarity normalization, identical match should score at top (~1.0)
    assert scores[0] == pytest.approx(1.0)
    result = retriever.retrieve(RetrievalRequest(query="Doc B", top_k=1, mode="dense"))
    docs, scores = result.documents, result.scores
    assert docs and docs[0].content == "Doc B"


def test_sparse_bm25_retriever_basic():
    # Sin tokenización real, pero BM25Okapi exige listas de palabras, así que "hackeamos":

    class DummyBM25:
        def __init__(self):
            pass

        def get_scores(self, query):
            # Returns 1.0 if "a" is in query, 0.0 if not.
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
    result = retriever.retrieve(RetrievalRequest(query="a", top_k=1, mode="sparse"))
    docs, scores = result.documents, result.scores
    assert len(docs) == 1 and docs[0].content == "Doc A"
    assert scores[0] == 1.0

    result = retriever.retrieve(RetrievalRequest(query="b", top_k=1, mode="sparse"))
    docs, scores = result.documents, result.scores
    assert len(docs) == 1 and docs[0].content == "Doc B"
    assert scores[0] == 1.0


def test_sparse_bm25_retriever_flat_scores_fall_back_to_zero():
    class DummyBM25:
        def get_scores(self, query):
            return [4.0, 4.0]

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
    result = retriever.retrieve(RetrievalRequest(query="flat", top_k=2, mode="sparse"))
    assert result.scores == (0.0, 0.0)


def test_hybrid_retriever_merges_and_ranks():
    # Dense and sparse retrievers produce 1 doc each, they are merged, both must appear in top-2
    class DummyRetriever:
        def __init__(self, docs, scores):
            self._docs, self._scores = docs, scores

        def retrieve(self, request):
            k = request.top_k
            return RetrievalResult(
                items=tuple(
                    RetrievedDoc(document=d, score=s, stage="test")
                    for d, s in zip(self._docs[:k], self._scores[:k], strict=False)
                ),
                mode_used="sparse",
                backend_used="test",
            )

    doc_a = Document(id=1, content="Doc A")
    doc_b = Document(id=2, content="Doc B")
    dense = DummyRetriever([doc_a], [0.7])
    sparse = DummyRetriever([doc_b], [1.0])
    hybrid = HybridRetriever(dense=dense, sparse=sparse, alpha=0.5)
    result = hybrid.retrieve(RetrievalRequest(query="irrelevant", top_k=2, mode="hybrid"))
    docs, scores = result.documents, result.scores
    assert {d.content for d in docs} == {"Doc A", "Doc B"}
    assert len(scores) == 2
