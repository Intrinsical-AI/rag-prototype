from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import ANY

import pytest

from local_rag_backend.app.composition import (
    DEFAULT_DENSE_BACKEND_MESSAGE,
    build_dense_embedder_from_settings,
    build_retriever_with_default_embedder_from_settings,
    resolve_preferred_llm_provider,
)
from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError


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

    assert out == "dense-retriever"
    assert seen["vector_kwargs"] == {
        "index_path": "index.faiss",
        "id_map_path": "id_map.json",
        "dim": 7,
    }
    assert seen["dense_kwargs"] == {"embedder": ANY, "faiss_index": "vec-repo", "doc_repo": doc_repo}


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
