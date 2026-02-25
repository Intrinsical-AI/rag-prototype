from __future__ import annotations

from contextlib import contextmanager

import pytest

from local_rag_backend.app import factory
from local_rag_backend.app.routers import index as index_router
from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_upsert_dense_vector_write_failure_triggers_rebuild_and_succeeds(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class DummyEmbedder:
        dim = 1

        def embed(self, texts):
            return [[1.0] for _ in texts]

    class FailingVec:
        def delete(self, ids):
            return None

        def upsert(self, ids, vectors):
            raise RuntimeError("vec upsert fail")

    rebuild_calls = 0

    def _rebuild(*, doc_repo, vec_repo, embedder):
        nonlocal rebuild_calls
        rebuild_calls += 1
        return len(doc_repo.get_all_documents())

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: DummyEmbedder(), raising=True)
    monkeypatch.setattr(factory, "VectorStorage", lambda *a, **k: FailingVec(), raising=True)
    monkeypatch.setattr(factory, "rebuild_index_from_db", _rebuild, raising=True)

    resp = await asgi_client.post(
        "/api/docs/upsert", json={"docs": [{"external_id": "doc-1", "content": "hello"}]}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["rebuilt_index"] is True
    assert rebuild_calls == 1


async def test_upsert_mutation_fails_when_write_lock_cannot_be_acquired(
    asgi_client, in_memory_sqlite, monkeypatch
):
    @contextmanager
    def _broken_lock():
        raise RuntimeError("lock unavailable")
        yield

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(factory, "multi_store_write_lock", _broken_lock, raising=True)

    with pytest.raises(RuntimeError, match="lock unavailable"):
        await asgi_client.post(
            "/api/docs/upsert", json={"docs": [{"external_id": "doc-1", "content": "hello"}]}
        )

    assert SqlDocumentStorage().get_all_documents() == []


async def test_upsert_crash_window_keeps_sql_when_vector_and_rebuild_fail(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class DummyEmbedder:
        dim = 1

        def embed(self, texts):
            return [[1.0] for _ in texts]

    class FailingVec:
        def delete(self, ids):
            return None

        def upsert(self, ids, vectors):
            raise RuntimeError("vec upsert fail")

    def _rebuild_fail(**_kwargs):
        raise RuntimeError("rebuild failed")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: DummyEmbedder(), raising=True)
    monkeypatch.setattr(factory, "VectorStorage", lambda *a, **k: FailingVec(), raising=True)
    monkeypatch.setattr(factory, "rebuild_index_from_db", _rebuild_fail, raising=True)

    with pytest.raises(RuntimeError, match="rebuild failed"):
        await asgi_client.post(
            "/api/docs/upsert", json={"docs": [{"external_id": "doc-1", "content": "hello"}]}
        )

    docs = SqlDocumentStorage().get_all_documents()
    assert len(docs) == 1
    assert docs[0].external_id == "doc-1"


async def test_index_rebuild_failure_still_invalidates_cached_rag_service(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class DummyEmbedder:
        dim = 1

        def embed(self, texts):
            return [[1.0] for _ in texts]

    class DummyVec:
        def __init__(self, *args, **kwargs):
            return None

    reset_calls = 0

    def _count_reset() -> None:
        nonlocal reset_calls
        reset_calls += 1

    def _rebuild_fail(**_kwargs):
        raise RuntimeError("rebuild failed")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: DummyEmbedder(), raising=True)
    monkeypatch.setattr(factory, "VectorStorage", lambda *a, **k: DummyVec(), raising=True)
    monkeypatch.setattr(factory, "rebuild_index_from_db", _rebuild_fail, raising=True)
    monkeypatch.setattr(index_router, "reset_rag_service", _count_reset, raising=True)

    with pytest.raises(RuntimeError, match="rebuild failed"):
        await asgi_client.post("/api/index/rebuild")

    assert reset_calls == 1
