from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever


class DummyDocRepo:
    def __init__(self, docs):
        self._docs = list(docs)

    def get(self, ids):
        wanted = {str(doc_id) for doc_id in ids}
        return [doc for doc in self._docs if str(doc.id) in wanted]

    def get_all_documents(self):
        return list(self._docs)


class DummyEmbedder:
    dim = 2

    def embed(self, texts):
        out = []
        for text in texts:
            normalized = text.lower()
            if "auth" in normalized:
                out.append([1.0, 0.0])
            elif "sql" in normalized:
                out.append([0.0, 1.0])
            else:
                out.append([0.5, 0.5])
        return out


class DummyVectorRepo:
    def similar(self, vector, k):
        if vector == [1.0, 0.0]:
            return [("doc-auth", 0.95), ("doc-sql", 0.40)]
        return [("doc-sql", 0.95), ("doc-auth", 0.40)]


def test_sparse_applies_filters_before_scoring():
    docs = [
        Document(
            id="doc-auth",
            content="Auth token check in Python",
            source_id="repo-a",
            metadata={"language": "python", "unit_type": "function"},
        ),
        Document(
            id="doc-js",
            content="Auth token check in JavaScript",
            source_id="repo-b",
            metadata={"language": "javascript", "unit_type": "function"},
        ),
    ]
    retriever = LocalSplitSearchRetriever(doc_repo=DummyDocRepo(docs), preloaded_docs=docs)
    result = retriever.retrieve(
        RetrievalRequest(
            query="auth token",
            top_k=2,
            mode="sparse",
            filters=(RetrievalFilter(field="language", values=("python",)),),
        )
    )
    assert [item.document.id for item in result.items] == ["doc-auth"]
    assert result.mode_used == "sparse"


def test_dual_reranks_only_sparse_candidates():
    docs = [
        Document(
            id="doc-auth",
            content="Python auth guard",
            source_id="repo-a",
            metadata={"language": "python", "unit_type": "function"},
        ),
        Document(
            id="doc-sql",
            content="SQL escaping helper",
            source_id="repo-a",
            metadata={"language": "python", "unit_type": "function"},
        ),
    ]
    retriever = LocalSplitSearchRetriever(
        doc_repo=DummyDocRepo(docs),
        preloaded_docs=docs,
        embedder=DummyEmbedder(),
        vector_repo=DummyVectorRepo(),
    )
    result = retriever.retrieve(
        RetrievalRequest(query="auth bug", top_k=1, mode="dual", dual_candidate_k=2)
    )
    assert [item.document.id for item in result.items] == ["doc-auth"]
    assert result.mode_used == "dual"
    assert result.candidate_count == 2
