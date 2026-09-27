from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from local_rag_backend.composition.adapters import (
    DEFAULT_DENSE_BACKEND_MESSAGE,
    build_dense_embedder_from_settings,
    build_retriever_with_default_embedder_from_settings,
    resolve_preferred_llm_provider,
)
from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError, LLMConfigurationError
from local_rag_backend.infrastructure.embeddings.cached import ContentAddressedCachingEmbedder
from local_rag_backend.infrastructure.search_backends.elastic_like import ElasticLikeSearchRetriever
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever
from local_rag_backend.settings import Settings


class DummyEmbedder:
    dim = 4
    identity = EmbeddingIdentity("openai", "test-model", 4)

    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]


def _settings(**updates):
    return Settings(**{"openai_api_key": "k", "disable_embedding_cache": True, **updates})


def test_build_dense_embedder_uses_default_missing_backend_message():
    def unavailable(_model_name):
        raise RuntimeError("backend unavailable")

    with pytest.raises(EmbeddingsBackendUnavailableError, match="requires an embeddings backend"):
        build_dense_embedder_from_settings(
            settings_obj=_settings(openai_api_key=None), st_embedder_factory=unavailable
        )
    assert "dense-st" in DEFAULT_DENSE_BACKEND_MESSAGE


def test_build_dense_embedder_wraps_provider_with_persistent_cache(tmp_path):
    cfg = _settings(disable_embedding_cache=False, embedding_cache_db_path=str(tmp_path / "cache"))
    embedder = build_dense_embedder_from_settings(
        settings_obj=cfg, openai_embedder_factory=DummyEmbedder
    )
    assert isinstance(embedder, ContentAddressedCachingEmbedder)
    assert embedder.identity == DummyEmbedder.identity
    assert embedder.model_key == DummyEmbedder.identity.model_key


@pytest.mark.parametrize("backend", ["auto", "numpy"])
def test_build_retriever_passes_settings_identity_and_vector_backend(backend):
    cfg = _settings(vector_backend=backend)
    seen = {}

    def vector_repo_factory(**kwargs):
        seen.update(kwargs)
        return object()

    def unused_st(_model_name):
        raise AssertionError("OpenAI provider must be selected")

    retriever = build_retriever_with_default_embedder_from_settings(
        settings_obj=cfg,
        retrieval_mode="dense",
        doc_repo=SimpleNamespace(),
        openai_embedder_factory=DummyEmbedder,
        st_embedder_factory=unused_st,
        vector_repo_factory=vector_repo_factory,
    )
    assert isinstance(retriever, LocalSplitSearchRetriever)
    assert seen == {
        "index_path": cfg.index_path,
        "id_map_path": cfg.id_map_path,
        "dim": DummyEmbedder.dim,
        "backend": backend,
        "settings_obj": cfg,
        "embedding_identity": DummyEmbedder.identity,
    }


@pytest.mark.parametrize("backend, mode", [("elasticsearch", "sparse"), ("opensearch", "dual")])
def test_build_retriever_supports_remote_modes(backend, mode):
    cfg = _settings(search_backend=backend, es_base_url="http://es", os_base_url="http://os")
    retriever = build_retriever_with_default_embedder_from_settings(
        settings_obj=cfg,
        retrieval_mode=mode,
        doc_repo=SimpleNamespace(),
        openai_embedder_factory=DummyEmbedder,
    )
    assert isinstance(retriever, ElasticLikeSearchRetriever)
    assert retriever._backend_name == backend
    retriever._client.close()


def test_build_retriever_rejects_sparse_for_mismatched_elasticsearch_backend():
    cfg = _settings(
        persistence_backend="elasticsearch", es_base_url="http://es", retrieval_mode="hybrid"
    )
    with pytest.raises(ValueError, match="supports retrieval_mode=sparse only when"):
        build_retriever_with_default_embedder_from_settings(
            settings_obj=cfg, retrieval_mode="sparse", doc_repo=SimpleNamespace()
        )


@pytest.mark.parametrize("search_backend", ["elasticsearch", "local_split"])
def test_hybrid_uses_filtered_sparse_branch_for_selected_search_backend(search_backend):
    cfg = _settings(
        persistence_backend="elasticsearch",
        search_backend=search_backend,
        es_base_url="http://es",
        retrieval_mode="hybrid",
        hybrid_retrieval_alpha=0.25,
    )
    docs = [
        Document("public", "hello", metadata={"scope": "public"}),
        Document("other", "hello", metadata={"scope": "other"}),
    ]
    retriever = build_retriever_with_default_embedder_from_settings(
        settings_obj=cfg,
        retrieval_mode="hybrid",
        doc_repo=SimpleNamespace(
            get_all_documents=lambda: docs, get=lambda ids: [doc for doc in docs if doc.id in ids]
        ),
        openai_embedder_factory=DummyEmbedder,
        vector_repo_factory=lambda **_kwargs: SimpleNamespace(similar=lambda _vector, _k: []),
    )
    request = RetrievalRequest(
        query="hello", top_k=2, mode="hybrid", filters=(RetrievalFilter("scope", ("public",)),)
    )
    if search_backend == "local_split":
        assert isinstance(retriever.sparse, LocalSplitSearchRetriever)
        assert [doc.id for doc in retriever.retrieve(request).documents] == ["public"]
        return
    assert isinstance(retriever.sparse, ElasticLikeSearchRetriever)
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"hits": {"hits": []}})

    retriever.sparse._client.close()
    with httpx.Client(base_url="http://es", transport=httpx.MockTransport(respond)) as client:
        retriever.sparse._client = client
        retriever.retrieve(request)
    assert bodies[0]["query"]["bool"]["filter"] == [{"terms": {"scope": ["public"]}}]
    assert retriever.alpha == 0.25


@pytest.mark.parametrize(
    "openai_api_key, ollama_enabled, expected",
    [
        ("k", True, "ollama"),
        ("k", False, "openai"),
        (None, True, "ollama"),
    ],
)
def test_resolve_preferred_llm_provider(openai_api_key, ollama_enabled, expected):
    assert (
        resolve_preferred_llm_provider(
            settings_obj=_settings(
                openai_api_key=openai_api_key,
                ollama_enabled=ollama_enabled,
            )
        )
        == expected
    )


def test_resolve_preferred_llm_provider_raises_when_none_available():
    with pytest.raises(LLMConfigurationError, match="No LLM configured"):
        resolve_preferred_llm_provider(
            settings_obj=_settings(openai_api_key=None, ollama_enabled=False)
        )
