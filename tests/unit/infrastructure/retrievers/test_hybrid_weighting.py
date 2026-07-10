# tests/unit/infrastructure/retrievers/test_hybrid_weighting.py
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalRequest, RetrievalResult, RetrievedDoc
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever


class R:
    def __init__(self, doc, score):
        self.doc, self.score = doc, score

    def retrieve(self, request):
        return RetrievalResult(
            items=(RetrievedDoc(document=self.doc, score=self.score, stage="test"),),
            mode_used="sparse",
            backend_used="test",
        )


def test_hybrid_alpha_changes_ranking():
    a, b = Document(1, "A"), Document(2, "B")
    dense = R(a, 1.0)
    sparse = R(b, 0.99)
    req = RetrievalRequest(query="q", top_k=1, mode="hybrid")
    # alpha=1 → manda sparse
    h1 = HybridRetriever(dense=dense, sparse=sparse, alpha=1.0)
    result1 = h1.retrieve(req)
    assert result1.documents[0].id == 2
    # alpha=0 → manda dense
    h0 = HybridRetriever(dense=dense, sparse=sparse, alpha=0.0)
    result0 = h0.retrieve(req)
    assert result0.documents[0].id == 1
