from types import SimpleNamespace

import pytest
from support.container import override_container

from local_rag_backend.composition import factory
from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.http import dependencies as deps
from local_rag_backend.settings import get_settings


@pytest.mark.parametrize(
    "mode, provider",
    [
        ("sparse", "openai"),
        ("dense", "ollama"),
        ("hybrid", "openai"),
    ],
)
async def test_get_rag_service_uses_explicit_container_factories(
    mode,
    provider,
    monkeypatch,
    in_memory_sqlite,
):
    settings = get_settings()
    monkeypatch.setattr(settings, "retrieval_mode", mode)
    monkeypatch.setattr(settings, "openai_api_key", "k" if provider == "openai" else None)
    monkeypatch.setattr(settings, "ollama_enabled", provider == "ollama")

    class DummyDocRepo:
        def get_all_documents(self):
            return [Document(id="1", content="doc1"), Document(id="2", content="doc2")]

    class DummyEmbedder:
        dim = 4
        identity = EmbeddingIdentity("sentence_transformers", "test", 4)

    generator = SimpleNamespace()
    override_container(
        monkeypatch,
        doc_repo_factory=DummyDocRepo,
        openai_embedder_factory=DummyEmbedder,
        st_embedder_factory=lambda _model: DummyEmbedder(),
        vector_repo_factory=lambda **_kwargs: SimpleNamespace(),
        openai_generator_factory=lambda **_kwargs: generator,
        ollama_generator_factory=lambda **_kwargs: generator,
    )
    first = await deps.get_rag_service()
    second = await deps.get_rag_service()
    assert first is second
    assert first.generator is generator


async def test_ingest_docs_resets_cached_rag_service(asgi_client, in_memory_sqlite, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")
    monkeypatch.setattr(settings, "openai_api_key", "k")
    monkeypatch.setattr(settings, "ollama_enabled", False)
    calls = []

    def build(_self):
        obj = object()
        calls.append(obj)
        return obj

    monkeypatch.setattr(factory.AppContainer, "build_rag_service", build)
    first = await deps.get_rag_service()
    assert first is calls[0]

    response = await asgi_client.post("/api/docs/ingest", json={"texts": ["hello"]})
    assert response.status_code == 200
    assert response.json()["count"] == 1
    second = await deps.get_rag_service()
    assert second is calls[1]
    assert second is not first


async def test_app_context_and_settings_dependencies_share_runtime_context(in_memory_sqlite):
    context = deps.get_app_context()
    assert await deps.get_settings_dependency() is context.settings
    assert await deps.get_app_container_dependency() is context.container
