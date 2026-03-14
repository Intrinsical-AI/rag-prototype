from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_rag_backend.composition.adapters import (
    DEFAULT_DENSE_BACKEND_MESSAGE,
    build_dense_embedder_from_settings,
    build_retriever_with_default_embedder_from_settings,
    resolve_preferred_llm_provider,
)
from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError
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
        return SimpleNamespace(retrieve=lambda _query, _k=5: ([], []))

    def _hybrid_retriever_factory(**kwargs):
        seen["hybrid_kwargs"] = kwargs
        kwargs["sparse"].retrieve("hello", 3)
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
    with pytest.raises(RuntimeError, match="No LLM configured"):
        resolve_preferred_llm_provider(settings_obj=cfg)
