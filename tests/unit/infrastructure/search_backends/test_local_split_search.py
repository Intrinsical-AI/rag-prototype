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
    def __init__(self):
        self.calls: list[tuple[list[float], int]] = []

    def similar(self, vector, k):
        self.calls.append((list(vector), int(k)))
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


def test_dense_applies_filters_min_score_and_candidate_k_floor():
    docs = [
        Document(
            id="doc-auth",
            content="Python auth guard",
            source_id="repo-a",
            metadata={"language": "python"},
        ),
        Document(
            id="doc-sql",
            content="SQL escaping helper",
            source_id="repo-a",
            metadata={"language": "sql"},
        ),
    ]
    vector_repo = DummyVectorRepo()
    retriever = LocalSplitSearchRetriever(
        doc_repo=DummyDocRepo(docs),
        embedder=DummyEmbedder(),
        vector_repo=vector_repo,
        preloaded_docs=docs,
    )

    result = retriever.retrieve(
        RetrievalRequest(
            query="auth issue",
            top_k=2,
            candidate_k=1,
            mode="dense",
            min_score=0.5,
            filters=(RetrievalFilter(field="language", values=("python",)),),
        )
    )

    assert vector_repo.calls == [([1.0, 0.0], 2)]
    assert [item.document.id for item in result.items] == ["doc-auth"]
    assert result.mode_used == "dense"


def test_dense_returns_empty_when_vector_repo_has_no_candidates():
    class EmptyVectorRepo:
        def similar(self, _vector, _k):
            return []

    retriever = LocalSplitSearchRetriever(
        doc_repo=DummyDocRepo([]),
        embedder=DummyEmbedder(),
        vector_repo=EmptyVectorRepo(),
        preloaded_docs=[],
    )

    result = retriever.retrieve(RetrievalRequest(query="auth", top_k=2, mode="dense"))

    assert result.items == ()
    assert result.mode_used == "dense"


def test_sparse_supports_metadata_prefixed_filters():
    docs = [
        Document(
            id="doc-node",
            content="AP node seen by sensor alpha",
            source_id="tr3v0r:dataset:artifact:net_nodes",
            metadata={"doc_type": "net_node", "sensor_id": "sensor-alpha"},
        ),
        Document(
            id="doc-edge",
            content="Edge observed by sensor beta",
            source_id="tr3v0r:dataset:artifact:net_edges",
            metadata={"doc_type": "net_edge", "sensor_id": "sensor-beta"},
        ),
    ]
    retriever = LocalSplitSearchRetriever(doc_repo=DummyDocRepo(docs), preloaded_docs=docs)

    result = retriever.retrieve(
        RetrievalRequest(
            query="sensor",
            top_k=2,
            mode="sparse",
            filters=(
                RetrievalFilter(field="metadata.doc_type", values=("net_node",)),
                RetrievalFilter(field="metadata.sensor_id", values=("sensor-alpha",)),
            ),
        )
    )

    assert [item.document.id for item in result.items] == ["doc-node"]


def test_sparse_supports_membership_filters_for_metadata_sequences():
    docs = [
        Document(
            id="doc-node",
            content="AP node seen by sensors alpha and beta",
            source_id="tr3v0r:dataset:artifact:net_nodes",
            metadata={
                "doc_type": "net_node",
                "sensor_ids": ["sensor-alpha", "sensor-beta"],
            },
        ),
        Document(
            id="doc-edge",
            content="Edge observed by sensor gamma",
            source_id="tr3v0r:dataset:artifact:net_edges",
            metadata={"doc_type": "net_edge", "sensor_ids": ["sensor-gamma"]},
        ),
    ]
    retriever = LocalSplitSearchRetriever(doc_repo=DummyDocRepo(docs), preloaded_docs=docs)

    result = retriever.retrieve(
        RetrievalRequest(
            query="sensor",
            top_k=2,
            mode="sparse",
            filters=(
                RetrievalFilter(field="metadata.doc_type", values=("net_node",)),
                RetrievalFilter(field="metadata.sensor_ids", values=("sensor-beta",)),
            ),
        )
    )

    assert [item.document.id for item in result.items] == ["doc-node"]


def test_dense_requires_embedder_and_vector_repo():
    retriever = LocalSplitSearchRetriever(doc_repo=DummyDocRepo([]), preloaded_docs=[])

    try:
        retriever.retrieve(RetrievalRequest(query="auth", top_k=1, mode="dense"))
    except RuntimeError as exc:
        assert "embedder and vector_repo" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_dual_candidate_floor_uses_top_k_when_dual_candidate_k_is_smaller():
    docs = [
        Document(id="doc-auth", content="Python auth guard", source_id="repo-a"),
        Document(id="doc-sql", content="SQL escaping helper", source_id="repo-a"),
    ]
    retriever = LocalSplitSearchRetriever(
        doc_repo=DummyDocRepo(docs),
        preloaded_docs=docs,
        embedder=DummyEmbedder(),
        vector_repo=DummyVectorRepo(),
    )

    result = retriever.retrieve(
        RetrievalRequest(query="auth bug", top_k=2, mode="dual", dual_candidate_k=1)
    )

    assert [item.document.id for item in result.items] == ["doc-auth", "doc-sql"]


def test_legacy_retrieve_path_returns_docs_and_scores():
    docs = [
        Document(id="doc-auth", content="Python auth guard", source_id="repo-a"),
        Document(id="doc-sql", content="SQL escaping helper", source_id="repo-a"),
    ]
    retriever = LocalSplitSearchRetriever(doc_repo=DummyDocRepo(docs), preloaded_docs=docs)

    out_docs, out_scores = retriever.retrieve("auth", 1)

    assert [doc.id for doc in out_docs] == ["doc-auth"]
    assert len(out_scores) == 1
