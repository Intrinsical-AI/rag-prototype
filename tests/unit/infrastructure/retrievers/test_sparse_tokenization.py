# tests/unit/infrastructure/retrievers/test_sparse_tokenization.py
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


class Repo:
    def __init__(self):
        self.docs = [Document(1, "<b>Hola</b> mundo"), Document(2, "Adios...")]

    def get(self, ids):
        return [d for d in self.docs if d.id in ids]


def test_sparse_preprocess_and_norm(monkeypatch):
    # Force BM25 real to be present; if not, simulate
    try:
        import rank_bm25  # noqa
    except Exception:

        class DummyBM25:
            def __init__(self, _):
                pass

            def get_scores(self, q):
                return [1.0 if "hola" in q else 0.2, 0.0]

        monkeypatch.setattr(
            "local_rag_backend.infrastructure.retrieval.sparse_bm25.BM25Okapi",
            DummyBM25,
            raising=False,
        )

    retr = SparseBM25Retriever(
        documents=["<b>Hola</b> mundo", "Adios..."], doc_ids=[1, 2], doc_repo=Repo()
    )
    docs, scores = retr.retrieve("HOLA", k=1)
    assert docs and docs[0].id == 1
    assert 0.0 <= scores[0] <= 1.0
