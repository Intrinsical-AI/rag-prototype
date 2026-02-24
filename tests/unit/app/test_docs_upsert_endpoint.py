# tests/unit/app/test_docs_upsert_endpoint.py

import numpy as np

from local_rag_backend.app import factory
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_upsert_docs_sparse_is_idempotent(asgi_client, in_memory_sqlite, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    payload = {"docs": [{"external_id": "doc-1", "content": "hello"}]}
    r1 = await asgi_client.post("/api/docs/upsert", json=payload)
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["inserted"] == 1
    assert data1["updated"] == 0
    assert data1["unchanged"] == 0
    doc_id = data1["results"][0]["id"]

    # Same payload: unchanged, no new rows
    r2 = await asgi_client.post("/api/docs/upsert", json=payload)
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["inserted"] == 0
    assert data2["updated"] == 0
    assert data2["unchanged"] == 1
    assert data2["results"][0]["id"] == doc_id

    # Update content
    r3 = await asgi_client.post(
        "/api/docs/upsert", json={"docs": [{"external_id": "doc-1", "content": "hello2"}]}
    )
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3["inserted"] == 0
    assert data3["updated"] == 1
    assert data3["unchanged"] == 0
    assert data3["results"][0]["id"] == doc_id

    doc = SqlDocumentStorage().get([doc_id])[0]
    assert doc.content == "hello2"


async def test_upsert_docs_rejects_duplicate_external_ids(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    payload = {
        "docs": [
            {"external_id": "doc-1", "content": "a"},
            {"external_id": "doc-1", "content": "b"},
        ]
    }
    r = await asgi_client.post("/api/docs/upsert", json=payload)
    assert r.status_code == 400


async def test_upsert_docs_dense_updates_only_changed_content(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class DummyEmbedder:
        dim = 4

        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return np.zeros((len(texts), self.dim), dtype="float32").tolist()

    class DummyVec:
        def __init__(self):
            self.upserts = []
            self.deletes = []

        def upsert(self, ids, vectors):
            self.upserts.append((list(ids), list(vectors)))

        def delete(self, ids):
            self.deletes.append(list(ids))

        def rebuild(self, ids, vectors):
            raise AssertionError("rebuild should not be called in this test")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)

    dummy_embedder = DummyEmbedder()
    dummy_vec = DummyVec()

    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: dummy_embedder)
    monkeypatch.setattr(factory, "FaissVectorStorage", lambda *a, **k: dummy_vec)

    payload = {"docs": [{"external_id": "doc-1", "content": "hello"}]}
    r1 = await asgi_client.post("/api/docs/upsert", json=payload)
    assert r1.status_code == 200
    doc_id = r1.json()["results"][0]["id"]
    assert dummy_embedder.calls == 1
    assert dummy_vec.deletes == []
    assert len(dummy_vec.upserts) == 1
    assert dummy_vec.upserts[0][0] == [doc_id]

    # Idempotent: no embedding, no index operations
    r2 = await asgi_client.post("/api/docs/upsert", json=payload)
    assert r2.status_code == 200
    assert dummy_embedder.calls == 1
    assert len(dummy_vec.upserts) == 1
    assert dummy_vec.deletes == []

    # Content change: delete old vector then upsert new
    r3 = await asgi_client.post(
        "/api/docs/upsert", json={"docs": [{"external_id": "doc-1", "content": "hello2"}]}
    )
    assert r3.status_code == 200
    assert r3.json()["results"][0]["id"] == doc_id
    assert dummy_embedder.calls == 2
    assert dummy_vec.deletes == [[doc_id]]
    assert len(dummy_vec.upserts) == 2
