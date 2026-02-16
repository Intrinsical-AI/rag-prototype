# tests/unit/app/test_docs_delete_by_external_id.py

from __future__ import annotations

import numpy as np

from local_rag_backend.app import api_router as api
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_delete_by_external_id_sparse_delete_does_not_reappear_on_reingest_or_ask_eval(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    r1 = await asgi_client.post("/api/docs", json={"texts": ["hello world"]})
    assert r1.status_code == 200
    assert r1.json()["count"] == 1

    doc = SqlDocumentStorage().get_all_documents()[0]
    assert doc.external_id

    # Pre-delete: ask_eval should retrieve a source.
    class DummyGen:
        def __init__(self, *a, **k):
            pass

        def generate(self, question, contexts):
            return "ans"

    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(api, "OpenAIGenerator", lambda **k: DummyGen(), raising=True)

    r_eval1 = await asgi_client.post(
        "/api/ask_eval", json={"question": "hello", "config": {"retrieval_mode": "sparse", "k": 1}}
    )
    assert r_eval1.status_code == 200
    assert len(r_eval1.json()["sources"]) == 1

    # Delete by external_id (creates tombstone).
    rd = await asgi_client.post(
        "/api/docs/delete_by_external_id", json={"external_ids": [doc.external_id]}
    )
    assert rd.status_code == 200
    assert rd.json()["deleted_sql"] == 1
    assert rd.json()["tombstoned"] == 1

    assert SqlDocumentStorage().get_all_documents() == []

    # Re-ingest same text: should not reappear due to tombstone.
    r2 = await asgi_client.post("/api/docs", json={"texts": ["hello world"]})
    assert r2.status_code == 200
    assert r2.json()["count"] == 0
    assert SqlDocumentStorage().get_all_documents() == []

    # Post-delete: ask_eval should return no sources.
    r_eval2 = await asgi_client.post(
        "/api/ask_eval", json={"question": "hello", "config": {"retrieval_mode": "sparse", "k": 1}}
    )
    assert r_eval2.status_code == 200
    assert r_eval2.json()["sources"] == []


async def test_delete_by_external_id_dense_is_consistent_and_survives_rebuild(
    asgi_client, in_memory_sqlite, tmp_path, monkeypatch
):
    # Dense mode with dummy embedder + dummy vector store.
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "idx.npy"), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(tmp_path / "id_map.json"), raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    class DummyEmbedder:
        dim = 2

        def embed(self, texts):
            return np.zeros((len(texts), self.dim), dtype="float32").tolist()

    class DummyVec:
        def __init__(self):
            self.upserts: list[list[int]] = []
            self.deletes: list[list[int]] = []
            self.rebuilds: list[list[int]] = []
            self.ids: set[int] = set()

        def upsert(self, ids, vectors):
            self.upserts.append(list(ids))
            self.ids |= set(ids)

        def delete(self, ids):
            self.deletes.append(list(ids))
            self.ids -= set(ids)

        def rebuild(self, ids, vectors):
            self.rebuilds.append(list(ids))
            self.ids = set(ids)

        def similar(self, vector, k):
            return []

    dummy_vec = DummyVec()
    monkeypatch.setattr(
        api, "SentenceTransformerEmbedder", lambda **k: DummyEmbedder(), raising=True
    )
    monkeypatch.setattr(api, "FaissVectorStorage", lambda **k: dummy_vec, raising=True)

    r1 = await asgi_client.post("/api/docs", json={"texts": ["alpha", "beta"]})
    assert r1.status_code == 200
    assert r1.json()["count"] == 2
    assert len(dummy_vec.upserts) == 1

    docs = SqlDocumentStorage().get_all_documents()
    assert len(docs) == 2
    ext_to_id = {d.external_id: d.id for d in docs}

    victim_ext = next(iter(ext_to_id.keys()))
    victim_id = ext_to_id[victim_ext]

    rd = await asgi_client.post(
        "/api/docs/delete_by_external_id", json={"external_ids": [victim_ext]}
    )
    assert rd.status_code == 200
    assert rd.json()["deleted_sql"] == 1
    assert victim_id in dummy_vec.deletes[-1]

    # Rebuild index: deleted doc must not reappear.
    rr = await asgi_client.post("/api/index/rebuild", json={})
    assert rr.status_code == 200
    remaining = SqlDocumentStorage().get_all_documents()
    assert len(remaining) == 1
    assert remaining[0].id in dummy_vec.ids
    assert victim_id not in dummy_vec.ids


async def test_delete_by_external_id_deduplicates_request_values(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    r1 = await asgi_client.post(
        "/api/docs/upsert", json={"docs": [{"external_id": "doc-1", "content": "x"}]}
    )
    assert r1.status_code == 200

    rd = await asgi_client.post(
        "/api/docs/delete_by_external_id",
        json={"external_ids": [" doc-1 ", "doc-1", "doc-1"]},
    )
    assert rd.status_code == 200
    payload = rd.json()
    assert payload["deleted_sql"] == 1
    assert payload["tombstoned"] == 1
