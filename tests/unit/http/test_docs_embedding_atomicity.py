import pytest
from support.container import override_container

from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import get_settings

settings = get_settings()


async def test_docs_dense_embed_failure_does_not_persist_sql(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class BadEmbedder:
        dim = 4
        identity = EmbeddingIdentity(provider="openai", model="test", dimension=dim)

        def embed(self, texts):
            raise RuntimeError("embed fail")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    override_container(monkeypatch, openai_embedder_factory=lambda *a, **k: BadEmbedder())

    with pytest.raises(RuntimeError, match="embed fail"):
        await asgi_client.post("/api/docs/ingest", json={"texts": ["hello world"]})
    assert SqlDocumentStorage(in_memory_sqlite).get_all_documents() == []


async def test_upsert_dense_embed_failure_does_not_persist_sql(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class BadEmbedder:
        dim = 4
        identity = EmbeddingIdentity(provider="openai", model="test", dimension=dim)

        def embed(self, texts):
            raise RuntimeError("embed fail")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    override_container(monkeypatch, openai_embedder_factory=lambda *a, **k: BadEmbedder())

    with pytest.raises(RuntimeError, match="embed fail"):
        await asgi_client.post(
            "/api/docs/mutate",
            json={"upserts": [{"external_id": "doc-1", "content": "hello"}]},
        )
    assert SqlDocumentStorage(in_memory_sqlite).get_all_documents() == []
