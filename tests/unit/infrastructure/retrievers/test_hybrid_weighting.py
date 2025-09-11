# tests/unit/infrastructure/retrievers/test_hybrid_weighting.py
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever


class R:
    def __init__(self, doc, score):
        self.doc, self.score = doc, score

    def retrieve(self, q, k):
        return [self.doc], [self.score]


def test_hybrid_alpha_changes_ranking():
    a, b = Document(1, "A"), Document(2, "B")
    dense = R(a, 1.0)
    sparse = R(b, 0.99)
    # alpha=1 → manda sparse
    h1 = HybridRetriever(dense=dense, sparse=sparse, alpha=1.0)
    docs1, _ = h1.retrieve("q", k=1)
    assert docs1[0].id == 2
    # alpha=0 → manda dense
    h0 = HybridRetriever(dense=dense, sparse=sparse, alpha=0.0)
    docs0, _ = h0.retrieve("q", k=1)
    assert docs0[0].id == 1
