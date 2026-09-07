from __future__ import annotations

from collections.abc import Sequence

import pytest

import local_rag_backend.infrastructure.search_backends.local_split as local_split
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever


class DummyDocRepo:
    def __init__(self, docs: Sequence[Document]) -> None:
        self._docs = list(docs)

    def store_documents(self, contents: Sequence[str]) -> Sequence[DocId]:
        raise NotImplementedError

    def delete_documents(self, ids: Sequence[DocId]) -> None:
        raise NotImplementedError

    def get(self, ids: Sequence[DocId]) -> Sequence[Document]:
        wanted = {str(doc_id) for doc_id in ids}
        return [doc for doc in self._docs if str(doc.id) in wanted]

    def get_all_documents(self) -> Sequence[Document]:
        return list(self._docs)


class DummyEmbedder:
    dim = 2

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
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


def test_sparse_candidate_count_includes_examined_docs_beyond_top_k() -> None:
    docs = [
        Document(id=DocId(str(i)), content=f"authentication token example {i}") for i in range(5)
    ]
    result = LocalSplitSearchRetriever(doc_repo=DummyDocRepo(docs)).retrieve(
        RetrievalRequest(query="authentication", top_k=2, mode="sparse")
    )
    assert len(result.items) == 2
    assert result.candidate_count == 5


class DummyVectorRepo:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], int]] = []

    @property
    def ntotal(self) -> int:
        return 0

    def upsert(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    def apply_delta_atomic(
        self,
        *,
        delete_ids: Sequence[DocId],
        upserts: Sequence[tuple[DocId, Sequence[float]]],
    ) -> None:
        raise NotImplementedError

    def delete(self, ids: Sequence[DocId]) -> int:
        raise NotImplementedError

    def rebuild(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    def similar(self, vector: Sequence[float], k: int) -> list[tuple[DocId, float]]:
        self.calls.append((list(vector), int(k)))
        if list(vector) == [1.0, 0.0]:
            return [(DocId("doc-auth"), 0.95), (DocId("doc-sql"), 0.40)]
        return [(DocId("doc-sql"), 0.95), (DocId("doc-auth"), 0.40)]


def test_sparse_applies_filters_before_scoring() -> None:
    docs = [
        Document(
            id=DocId("doc-auth"),
            content="Auth token check in Python",
            source_id="repo-a",
            metadata={"language": "python", "unit_type": "function"},
        ),
        Document(
            id=DocId("doc-js"),
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
            filters=(RetrievalFilter(field="metadata.language", values=("python",)),),
        )
    )
    assert [item.document.id for item in result.items] == ["doc-auth"]
    assert result.mode_used == "sparse"


def test_sparse_uses_cached_retriever_when_unfiltered(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [
        Document(id=DocId("doc-auth"), content="Auth token check in Python"),
        Document(id=DocId("doc-sql"), content="SQL escaping helper"),
    ]

    class CachedSparse:
        def __init__(self) -> None:
            self.calls = 0

        def retrieve(self, request: RetrievalRequest):
            self.calls += 1
            return SparseBM25Retriever(
                documents=[doc.content for doc in docs],
                doc_ids=[doc.id for doc in docs],
                doc_repo=DummyDocRepo(docs),
                preloaded_docs=docs,
            ).retrieve(request)

    cached = CachedSparse()

    def _explode(*args, **kwargs):
        raise AssertionError("SparseBM25Retriever should not be rebuilt for unfiltered requests")

    monkeypatch.setattr(local_split, "SparseBM25Retriever", _explode, raising=True)
    retriever = LocalSplitSearchRetriever(
        doc_repo=DummyDocRepo(docs),
        preloaded_docs=docs,
        cached_sparse_retriever=cached,  # type: ignore[arg-type]
    )

    result = retriever.retrieve(RetrievalRequest(query="auth token", top_k=1, mode="sparse"))

    assert cached.calls == 1
    assert [item.document.id for item in result.items] == ["doc-auth"]


def test_sparse_rebuilds_subset_retriever_when_filters_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = [
        Document(
            id=DocId("doc-auth"),
            content="Auth token check in Python",
            metadata={"language": "python"},
        ),
        Document(
            id=DocId("doc-js"),
            content="Auth token check in JavaScript",
            metadata={"language": "javascript"},
        ),
    ]
    seen: list[list[str]] = []
    real_sparse = local_split.SparseBM25Retriever

    class RecordingSparse(real_sparse):
        def __init__(self, *, documents, doc_ids, doc_repo, preloaded_docs=None):
            seen.append(list(doc_ids))
            super().__init__(
                documents=documents,
                doc_ids=doc_ids,
                doc_repo=doc_repo,
                preloaded_docs=preloaded_docs,
            )

    monkeypatch.setattr(local_split, "SparseBM25Retriever", RecordingSparse, raising=True)
    retriever = LocalSplitSearchRetriever(
        doc_repo=DummyDocRepo(docs),
        preloaded_docs=docs,
        cached_sparse_retriever=RecordingSparse(
            documents=[doc.content for doc in docs],
            doc_ids=[doc.id for doc in docs],
            doc_repo=DummyDocRepo(docs),
            preloaded_docs=docs,
        ),
    )

    result = retriever.retrieve(
        RetrievalRequest(
            query="auth token",
            top_k=2,
            mode="sparse",
            filters=(RetrievalFilter(field="metadata.language", values=("python",)),),
        )
    )

    assert seen[-1] == ["doc-auth"]
    assert [item.document.id for item in result.items] == ["doc-auth"]


def test_dual_reranks_only_sparse_candidates() -> None:
    docs = [
        Document(
            id=DocId("doc-auth"),
            content="Python auth guard",
            source_id="repo-a",
            metadata={"language": "python", "unit_type": "function"},
        ),
        Document(
            id=DocId("doc-sql"),
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


def test_dense_applies_filters_min_score_and_candidate_k_floor() -> None:
    docs = [
        Document(
            id=DocId("doc-auth"),
            content="Python auth guard",
            source_id="repo-a",
            metadata={"language": "python"},
        ),
        Document(
            id=DocId("doc-sql"),
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
            filters=(RetrievalFilter(field="metadata.language", values=("python",)),),
        )
    )

    assert vector_repo.calls == [([1.0, 0.0], 2)]
    assert [item.document.id for item in result.items] == ["doc-auth"]
    assert result.mode_used == "dense"


def test_dense_returns_empty_when_vector_repo_has_no_candidates() -> None:
    class EmptyVectorRepo:
        @property
        def ntotal(self) -> int:
            return 0

        def upsert(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
            raise NotImplementedError

        def apply_delta_atomic(
            self,
            *,
            delete_ids: Sequence[DocId],
            upserts: Sequence[tuple[DocId, Sequence[float]]],
        ) -> None:
            raise NotImplementedError

        def delete(self, ids: Sequence[DocId]) -> int:
            raise NotImplementedError

        def rebuild(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
            raise NotImplementedError

        def similar(self, _vector: Sequence[float], _k: int) -> list[tuple[DocId, float]]:
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


def test_sparse_supports_metadata_prefixed_filters() -> None:
    docs = [
        Document(
            id=DocId("doc-node"),
            content="AP node seen by sensor alpha",
            source_id="tr3v0r:dataset:artifact:net_nodes",
            metadata={"doc_type": "net_node", "sensor_id": "sensor-alpha"},
        ),
        Document(
            id=DocId("doc-edge"),
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


def test_sparse_supports_membership_filters_for_metadata_sequences() -> None:
    docs = [
        Document(
            id=DocId("doc-node"),
            content="AP node seen by sensors alpha and beta",
            source_id="tr3v0r:dataset:artifact:net_nodes",
            metadata={
                "doc_type": "net_node",
                "sensor_ids": ["sensor-alpha", "sensor-beta"],
            },
        ),
        Document(
            id=DocId("doc-edge"),
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


def test_dense_requires_embedder_and_vector_repo() -> None:
    retriever = LocalSplitSearchRetriever(doc_repo=DummyDocRepo([]), preloaded_docs=[])

    try:
        retriever.retrieve(RetrievalRequest(query="auth", top_k=1, mode="dense"))
    except RuntimeError as exc:
        assert "embedder and vector_repo" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_dual_candidate_floor_uses_top_k_when_dual_candidate_k_is_smaller() -> None:
    docs = [
        Document(id=DocId("doc-auth"), content="Python auth guard", source_id="repo-a"),
        Document(id=DocId("doc-sql"), content="SQL escaping helper", source_id="repo-a"),
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
