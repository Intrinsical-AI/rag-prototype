from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_rag_backend.composition.adapters import (
    DEFAULT_DENSE_BACKEND_MESSAGE,
    build_dense_embedder_from_settings,
    build_retriever_with_default_embedder_from_settings,
    resolve_preferred_llm_provider,
)
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalRequest, RetrievalResult
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError, LLMConfigurationError
from local_rag_backend.infrastructure.embeddings.cached import ContentAddressedCachingEmbedder
from local_rag_backend.infrastructure.search_backends.elastic_like import ElasticLikeSearchRetriever
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever


def test_build_dense_embedder_uses_default_missing_backend_message():
    cfg = SimpleNamespace(openai_api_key=None, st_embedding_model="all-MiniLM-L6-v2")

    def _st_fail(_model_name: str):
        raise RuntimeError("backend unavailable")

    with pytest.raises(
        EmbeddingsBackendUnavailableError,
        match="Dense/hybrid retrieval requires an embeddings backend",
    ):
        build_dense_embedder_from_settings(
            settings_obj=cfg,
            openai_embedder_factory=lambda: object(),
            st_embedder_factory=_st_fail,
        )

    assert "dense-st" in DEFAULT_DENSE_BACKEND_MESSAGE


def test_build_dense_embedder_wraps_provider_with_persistent_cache(tmp_path):
    cfg = SimpleNamespace(
        openai_api_key="k",
        st_embedding_model="all-MiniLM-L6-v2",
        data_dir=tmp_path,
    )

    class DummyEmbedder:
        dim = 7
        model = "text-embedding-3-small"
        client = object()

        def embed(self, texts):
            return [[0.0] * self.dim for _ in texts]

    out = build_dense_embedder_from_settings(
        settings_obj=cfg,
        openai_embedder_factory=lambda: DummyEmbedder(),
        st_embedder_factory=lambda _model_name: DummyEmbedder(),
    )

    assert isinstance(out, ContentAddressedCachingEmbedder)
    assert out.model_key == "openai:text-embedding-3-small:7"


def test_build_retriever_with_default_embedder_uses_openai_factory_when_key_present():
    cfg = SimpleNamespace(
        openai_api_key="k",
        st_embedding_model="all-MiniLM-L6-v2",
        hybrid_retrieval_alpha=0.5,
        enable_reranker=False,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
        index_path="index.faiss",
        id_map_path="id_map.json",
    )
    doc_repo = SimpleNamespace()
    seen: dict[str, object] = {}

    class DummyEmbedder:
        dim = 7

    def _openai_embedder():
        return DummyEmbedder()

    def _st_embedder(_model_name: str):
        raise AssertionError("ST factory must not be used when OPENAI_API_KEY is set")

    def _vector_repo_factory(**kwargs):
        seen["vector_kwargs"] = kwargs
        return "vec-repo"

    def _dense_retriever_factory(**kwargs):
        seen["dense_kwargs"] = kwargs
        return "dense-retriever"

    out = build_retriever_with_default_embedder_from_settings(
        settings_obj=cfg,
        retrieval_mode="dense",
        doc_repo=doc_repo,
        openai_embedder_factory=_openai_embedder,
        st_embedder_factory=_st_embedder,
        vector_repo_factory=_vector_repo_factory,
        dense_retriever_factory=_dense_retriever_factory,
    )

    assert isinstance(out, LocalSplitSearchRetriever)
    assert seen["vector_kwargs"] == {
        "index_path": "index.faiss",
        "id_map_path": "id_map.json",
        "dim": 7,
        "backend": "auto",
        "settings_obj": cfg,
    }


def test_build_retriever_passes_vector_backend_to_repo_factory():
    cfg = SimpleNamespace(
        openai_api_key="k",
        st_embedding_model="all-MiniLM-L6-v2",
        hybrid_retrieval_alpha=0.5,
        enable_reranker=False,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
        index_path="index.faiss",
        id_map_path="id_map.json",
        vector_backend="numpy",
    )
    seen: dict[str, object] = {}

    class DummyEmbedder:
        dim = 4

    def _vector_repo_factory(**kwargs):
        seen["kwargs"] = kwargs
        return object()

    build_retriever_with_default_embedder_from_settings(
        settings_obj=cfg,
        retrieval_mode="dense",
        doc_repo=SimpleNamespace(get=lambda _ids: []),
        openai_embedder_factory=lambda: DummyEmbedder(),
        st_embedder_factory=lambda _model_name: DummyEmbedder(),
        vector_repo_factory=_vector_repo_factory,
        dense_retriever_factory=lambda **kwargs: kwargs,
    )

    assert seen["kwargs"] == {
        "index_path": "index.faiss",
        "id_map_path": "id_map.json",
        "dim": 4,
        "backend": "numpy",
        "settings_obj": cfg,
    }


def test_build_retriever_allows_sparse_with_elasticsearch_search_backend():
    cfg = SimpleNamespace(
        persistence_backend="local_split",
        search_backend="elasticsearch",
        retrieval_mode="sparse",
        es_base_url="http://localhost:9200",
        es_docs_index="rag-docs",
        es_content_field="content",
        es_embedding_field="embedding",
        es_request_timeout_s=5.0,
        es_verify_tls=False,
        es_api_key=None,
        es_username=None,
        es_password=None,
        es_hybrid_vector_k=50,
        openai_api_key="k",
        st_embedding_model="all-MiniLM-L6-v2",
        hybrid_retrieval_alpha=0.5,
        enable_reranker=False,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
        index_path="index.faiss",
        id_map_path="id_map.json",
        vector_backend="auto",
    )

    out = build_retriever_with_default_embedder_from_settings(
        settings_obj=cfg,
        retrieval_mode="sparse",
        doc_repo=SimpleNamespace(get_all_documents=lambda: []),
        openai_embedder_factory=lambda: object(),
        st_embedder_factory=lambda _model_name: object(),
    )

    assert isinstance(out, ElasticLikeSearchRetriever)


def test_build_retriever_allows_opensearch_dual_mode():
    cfg = SimpleNamespace(
        persistence_backend="local_split",
        search_backend="opensearch",
        retrieval_mode="dual",
        os_base_url="http://localhost:9200",
        os_docs_index="rag-docs",
        os_content_field="content",
        os_embedding_field="embedding",
        os_request_timeout_s=5.0,
        os_verify_tls=False,
        os_api_key=None,
        os_username=None,
        os_password=None,
        os_dense_candidate_k=50,
        openai_api_key="k",
        st_embedding_model="all-MiniLM-L6-v2",
        hybrid_retrieval_alpha=0.5,
        enable_reranker=False,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
        index_path="index.faiss",
        id_map_path="id_map.json",
        vector_backend="auto",
    )

    class DummyEmbedder:
        dim = 4

    out = build_retriever_with_default_embedder_from_settings(
        settings_obj=cfg,
        retrieval_mode="dual",
        doc_repo=SimpleNamespace(get_all_documents=lambda: []),
        openai_embedder_factory=lambda: DummyEmbedder(),
        st_embedder_factory=lambda _model_name: DummyEmbedder(),
    )

    assert isinstance(out, ElasticLikeSearchRetriever)
    assert out._backend_name == "opensearch"


def test_build_retriever_rejects_sparse_for_elasticsearch_backend():
    cfg = SimpleNamespace(
        persistence_backend="elasticsearch",
        search_backend="local_split",
        retrieval_mode="sparse",
        es_base_url="http://localhost:9200",
        openai_api_key="k",
        st_embedding_model="all-MiniLM-L6-v2",
        hybrid_retrieval_alpha=0.5,
        enable_reranker=False,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
        index_path="index.faiss",
        id_map_path="id_map.json",
        vector_backend="auto",
    )

    with pytest.raises(ValueError, match="supports retrieval_mode=sparse only when"):
        build_retriever_with_default_embedder_from_settings(
            settings_obj=cfg,
            retrieval_mode="sparse",
            doc_repo=SimpleNamespace(get_all_documents=lambda: []),
            openai_embedder_factory=lambda: object(),
            st_embedder_factory=lambda _model_name: object(),
        )


def test_build_retriever_hybrid_uses_elasticsearch_lexical_path():
    cfg = SimpleNamespace(
        persistence_backend="elasticsearch",
        search_backend="elasticsearch",
        openai_api_key="k",
        st_embedding_model="all-MiniLM-L6-v2",
        hybrid_retrieval_alpha=0.25,
        enable_reranker=False,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
        index_path="index.faiss",
        id_map_path="id_map.json",
        vector_backend="auto",
    )
    seen: dict[str, object] = {}

    class DummyEmbedder:
        dim = 4

    class DummyVectorRepo:
        def similar(self, _vector, _k):
            return []

        def lexical_search(self, _query, *, k):
            seen["lexical_k"] = k
            return []

    def _dense_retriever_factory(**kwargs):
        seen["dense_kwargs"] = kwargs
        return SimpleNamespace(
            retrieve=lambda _req: RetrievalResult(items=(), mode_used="dense", backend_used="test")
        )

    def _hybrid_retriever_factory(**kwargs):
        seen["hybrid_kwargs"] = kwargs
        kwargs["sparse"].retrieve(RetrievalRequest(query="hello", top_k=3, mode="sparse"))
        return "hybrid-retriever"

    out = build_retriever_with_default_embedder_from_settings(
        settings_obj=cfg,
        retrieval_mode="hybrid",
        doc_repo=SimpleNamespace(get=lambda _ids: []),
        openai_embedder_factory=lambda: DummyEmbedder(),
        st_embedder_factory=lambda _model_name: DummyEmbedder(),
        vector_repo_factory=lambda **kwargs: DummyVectorRepo(),
        dense_retriever_factory=_dense_retriever_factory,
        hybrid_retriever_factory=_hybrid_retriever_factory,
    )

    assert out == "hybrid-retriever"
    assert seen["lexical_k"] == 3
    assert seen["dense_kwargs"]["doc_repo"] is not None
    assert seen["hybrid_kwargs"]["alpha"] == 0.25


def test_build_retriever_hybrid_uses_elasticsearch_lexical_path_for_local_split_search_backend():
    docs = [
        Document(id=DocId("doc-1"), content="capital france paris", external_id="doc-1"),
        Document(id=DocId("doc-2"), content="sql escaping helper", external_id="doc-2"),
    ]
    cfg = SimpleNamespace(
        persistence_backend="elasticsearch",
        search_backend="local_split",
        es_base_url="http://localhost:9200",
        openai_api_key="k",
        st_embedding_model="all-MiniLM-L6-v2",
        hybrid_retrieval_alpha=0.4,
        enable_reranker=False,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
        index_path="index.faiss",
        id_map_path="id_map.json",
        vector_backend="auto",
    )
    seen: dict[str, object] = {}

    class DummyEmbedder:
        dim = 4

    class DummyVectorRepo:
        def lexical_search(self, _query, *, k):
            seen["lexical_k"] = k
            return [(DocId("doc-1"), 0.91), (DocId("doc-2"), 0.12)]

    def _dense_retriever_factory(**kwargs):
        seen["dense_kwargs"] = kwargs
        return SimpleNamespace(
            retrieve=lambda _req: RetrievalResult(items=(), mode_used="dense", backend_used="dense")
        )

    def _sparse_retriever_factory(**_kwargs):
        raise AssertionError("sparse_retriever_factory must not be used for ES-backed hybrid")

    def _vector_repo_factory(**kwargs):
        seen["vector_kwargs"] = kwargs
        return DummyVectorRepo()

    def _hybrid_retriever_factory(**kwargs):
        seen["hybrid_kwargs"] = kwargs
        sparse_result = kwargs["sparse"].retrieve(RetrievalRequest(query="hello", top_k=2, mode="sparse"))
        seen["sparse_result"] = sparse_result
        return "hybrid-retriever"

    out = build_retriever_with_default_embedder_from_settings(
        settings_obj=cfg,
        retrieval_mode="hybrid",
        doc_repo=SimpleNamespace(
            get_all_documents=lambda: docs,
            get=lambda ids: [doc for doc in docs if doc.id in set(ids)],
        ),
        openai_embedder_factory=lambda: DummyEmbedder(),
        st_embedder_factory=lambda _model_name: DummyEmbedder(),
        sparse_retriever_factory=_sparse_retriever_factory,
        vector_repo_factory=_vector_repo_factory,
        dense_retriever_factory=_dense_retriever_factory,
        hybrid_retriever_factory=_hybrid_retriever_factory,
    )

    assert out == "hybrid-retriever"
    assert seen["vector_kwargs"] == {
        "index_path": "index.faiss",
        "id_map_path": "id_map.json",
        "dim": 4,
        "backend": "auto",
        "settings_obj": cfg,
    }
    assert seen["lexical_k"] == 2
    assert seen["sparse_result"].backend_used == "elastic_lexical"
    assert tuple(doc.id for doc in seen["sparse_result"].documents) == (DocId("doc-1"), DocId("doc-2"))
    assert seen["dense_kwargs"]["doc_repo"] is not None
    assert seen["hybrid_kwargs"]["alpha"] == 0.4


@pytest.mark.parametrize(
    ("openai_api_key", "ollama_enabled", "expected"),
    [
        ("k", True, "ollama"),
        ("k", False, "openai"),
        (None, True, "ollama"),
    ],
)
def test_resolve_preferred_llm_provider(openai_api_key, ollama_enabled, expected):
    cfg = SimpleNamespace(openai_api_key=openai_api_key, ollama_enabled=ollama_enabled)
    assert resolve_preferred_llm_provider(settings_obj=cfg) == expected


def test_resolve_preferred_llm_provider_raises_when_none_available():
    cfg = SimpleNamespace(openai_api_key=None, ollama_enabled=False)
    with pytest.raises(LLMConfigurationError, match="No LLM configured"):
        resolve_preferred_llm_provider(settings_obj=cfg)
