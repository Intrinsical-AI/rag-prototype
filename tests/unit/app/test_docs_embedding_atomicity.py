import pytest

from local_rag_backend.app import api_router as api
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
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
    monkeypatch.setattr(api, "OpenAIEmbedder", lambda *a, **k: BadEmbedder(), raising=True)

    with pytest.raises(RuntimeError, match="embed fail"):
        await asgi_client.post("/api/docs", json={"texts": ["hello world"]})
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
    monkeypatch.setattr(api, "OpenAIEmbedder", lambda *a, **k: BadEmbedder(), raising=True)

    with pytest.raises(RuntimeError, match="embed fail"):
        await asgi_client.post(
            "/api/docs/upsert", json={"docs": [{"external_id": "doc-1", "content": "hello"}]}
        )
    assert SqlDocumentStorage().get_all_documents() == []
