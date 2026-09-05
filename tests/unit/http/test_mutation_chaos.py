from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from local_rag_backend.composition import factory
from local_rag_backend.core.errors import WriteLockTimeoutError
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_mutate_dense_vector_failure_rolls_back_sql(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class DummyEmbedder:
        dim = 1

        def embed(self, texts):
            return [[1.0] for _ in texts]

    class FailingVec:
        def apply_delta_atomic(self, *, delete_ids, upserts):
            raise RuntimeError("vec upsert fail")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: DummyEmbedder(), raising=True)
    monkeypatch.setattr(factory, "VectorStorage", lambda *a, **k: FailingVec(), raising=True)

    with pytest.raises(RuntimeError, match="vec upsert fail"):
        await asgi_client.post(
            "/api/docs/mutate",
            json={"upserts": [{"external_id": "doc-1", "content": "hello"}]},
        )

    assert SqlDocumentStorage().get_all_documents() == []


async def test_mutation_fails_when_write_lock_unavailable_returns_503(
    asgi_client, in_memory_sqlite, monkeypatch
):
    @contextmanager
    def _broken_lock(*, coordination_dir=None, timeout_s=None, poll_s=None):
        raise WriteLockTimeoutError("lock unavailable")
        yield

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(factory, "multi_store_write_lock", _broken_lock, raising=True)
    factory.reset_app_context()

    try:
        resp = await asgi_client.post(
            "/api/docs/mutate",
            json={"upserts": [{"external_id": "doc-1", "content": "hello"}]},
        )
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Service unavailable."
        assert "lock unavailable" not in resp.text
        assert SqlDocumentStorage().get_all_documents() == []
    finally:
        factory.reset_app_context()


async def test_vector_failure_persists_rolled_back_journal_record(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class DummyEmbedder:
        dim = 1

        def embed(self, texts):
            return [[1.0] for _ in texts]

    class FailingVec:
        def apply_delta_atomic(self, *, delete_ids, upserts):
            raise RuntimeError("vec upsert fail")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: DummyEmbedder(), raising=True)
    monkeypatch.setattr(factory, "VectorStorage", lambda *a, **k: FailingVec(), raising=True)

    journal_dir = Path(settings.get_coordination_dir()) / ".mutation_journal"
    before = sorted(journal_dir.glob("*.json")) if journal_dir.is_dir() else []

    with pytest.raises(RuntimeError, match="vec upsert fail"):
        await asgi_client.post(
            "/api/docs/mutate",
            json={"upserts": [{"external_id": "doc-journal", "content": "x"}]},
        )

    after = sorted(journal_dir.glob("*.json")) if journal_dir.is_dir() else []
    assert len(after) >= len(before) + 1
    last_record = json.loads(after[-1].read_text(encoding="utf-8"))
    assert last_record.get("state") == "ROLLED_BACK"
