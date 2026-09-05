import pytest

from local_rag_backend.composition import factory
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_docs_dense_embed_failure_does_not_persist_sql(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class BadEmbedder:
        dim = 4

        def embed(self, texts):
            raise RuntimeError("embed fail")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: BadEmbedder(), raising=True)

    with pytest.raises(RuntimeError, match="embed fail"):
        await asgi_client.post("/api/docs/ingest", json={"texts": ["hello world"]})
    assert SqlDocumentStorage().get_all_documents() == []


async def test_upsert_dense_embed_failure_does_not_persist_sql(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class BadEmbedder:
        dim = 4

        def embed(self, texts):
            raise RuntimeError("embed fail")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: BadEmbedder(), raising=True)

    with pytest.raises(RuntimeError, match="embed fail"):
        await asgi_client.post(
            "/api/docs/mutate",
            json={"upserts": [{"external_id": "doc-1", "content": "hello"}]},
        )
    assert SqlDocumentStorage().get_all_documents() == []
