from __future__ import annotations

import numpy as np

from local_rag_backend.composition import factory
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_mutate_delete_by_external_id_sparse_creates_tombstone_and_blocks_reingest(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    r1 = await asgi_client.post("/api/docs/ingest", json={"texts": ["hello world"]})
    assert r1.status_code == 200
    assert r1.json()["count"] == 1

    doc = SqlDocumentStorage().get_all_documents()[0]
    rd = await asgi_client.post(
        "/api/docs/mutate",
        json={"delete_external_ids": [doc.external_id]},
    )
    assert rd.status_code == 200
    assert rd.json()["deleted_sql"] == 1
    assert rd.json()["tombstoned"] == 1
    assert SqlDocumentStorage().get_all_documents() == []

    r2 = await asgi_client.post("/api/docs/ingest", json={"texts": ["hello world"]})
    assert r2.status_code == 200
    assert r2.json()["count"] == 0
    assert SqlDocumentStorage().get_all_documents() == []


async def test_mutate_upsert_rejects_tombstoned_external_id(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    r1 = await asgi_client.post(
        "/api/docs/mutate",
        json={"upserts": [{"external_id": "doc-tombstone-1", "content": "hello"}]},
    )
    assert r1.status_code == 200

    r2 = await asgi_client.post(
        "/api/docs/mutate",
        json={"delete_external_ids": ["doc-tombstone-1"]},
    )
    assert r2.status_code == 200
    assert r2.json()["deleted_sql"] == 1
    assert r2.json()["tombstoned"] == 1

    r3 = await asgi_client.post(
        "/api/docs/mutate",
        json={"upserts": [{"external_id": "doc-tombstone-1", "content": "resurrect"}]},
    )
    assert r3.status_code == 400
    assert "tombstoned" in r3.json()["detail"].lower()
    assert SqlDocumentStorage().get_all_documents() == []


async def test_mutate_delete_by_external_id_dense_is_consistent_and_survives_rebuild(
    asgi_client, in_memory_sqlite, tmp_path, monkeypatch
):
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
            self.ops: list[dict[str, list[str]]] = []
            self.ids: set[str] = set()

        def apply_delta_atomic(self, *, delete_ids, upserts):
            delete_s = [str(x) for x in delete_ids]
            upsert_s = [str(doc_id) for doc_id, _ in upserts]
            self.ops.append({"delete_ids": delete_s, "upserts": upsert_s})
            self.ids -= set(delete_s)
            self.ids |= set(upsert_s)

        def rebuild(self, ids, vectors):
            self.ids = {str(x) for x in ids}

        def similar(self, vector, k):
            return []

    dummy_vec = DummyVec()
    monkeypatch.setattr(
        factory, "SentenceTransformerEmbedder", lambda **k: DummyEmbedder(), raising=True
    )
    monkeypatch.setattr(factory, "VectorStorage", lambda **k: dummy_vec, raising=True)

    r1 = await asgi_client.post("/api/docs/ingest", json={"texts": ["alpha", "beta"]})
    assert r1.status_code == 200
    assert r1.json()["count"] == 2
    assert len(dummy_vec.ops) == 1

    docs = SqlDocumentStorage().get_all_documents()
    ext_to_id = {d.external_id: str(d.id) for d in docs}
    victim_ext = next(iter(ext_to_id.keys()))
    victim_id = ext_to_id[victim_ext]

    rd = await asgi_client.post(
        "/api/docs/mutate",
        json={"delete_external_ids": [victim_ext]},
    )
    assert rd.status_code == 200
    assert rd.json()["deleted_sql"] == 1
    assert victim_id in dummy_vec.ops[-1]["delete_ids"]

    rr = await asgi_client.post("/api/index/rebuild", json={})
    assert rr.status_code == 200
    remaining = SqlDocumentStorage().get_all_documents()
    assert len(remaining) == 1
    assert str(remaining[0].id) in dummy_vec.ids
    assert victim_id not in dummy_vec.ids


async def test_mutate_delete_by_external_id_deduplicates_values(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    r1 = await asgi_client.post(
        "/api/docs/mutate",
        json={"upserts": [{"external_id": "doc-1", "content": "x"}]},
    )
    assert r1.status_code == 200

    rd = await asgi_client.post(
        "/api/docs/mutate",
        json={"delete_external_ids": [" doc-1 ", "doc-1", "doc-1"]},
    )
    assert rd.status_code == 200
    payload = rd.json()
    assert payload["deleted_sql"] == 1
    assert payload["tombstoned"] == 1


async def test_mutate_delete_by_external_id_dense_does_not_require_embedder(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    r1 = await asgi_client.post(
        "/api/docs/mutate",
        json={"upserts": [{"external_id": "doc-embedder-not-needed", "content": "hello"}]},
    )
    assert r1.status_code == 200

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    embedder_calls = 0

    def _boom_embedder(**_kwargs):
        nonlocal embedder_calls
        embedder_calls += 1
        raise RuntimeError("embedder should not be called on successful delete path")

    class DummyVec:
        def apply_delta_atomic(self, *, delete_ids, upserts):
            assert len(list(delete_ids)) == 1
            assert list(upserts) == []

    monkeypatch.setattr(factory, "SentenceTransformerEmbedder", _boom_embedder, raising=True)
    monkeypatch.setattr(factory, "VectorStorage", lambda **_k: DummyVec(), raising=True)

    rd = await asgi_client.post(
        "/api/docs/mutate",
        json={"delete_external_ids": ["doc-embedder-not-needed"]},
    )
    assert rd.status_code == 200
    payload = rd.json()
    assert payload["deleted_sql"] == 1
    assert payload["deleted_index"] == 1
    assert payload["index_rebuilt"] is False
    assert embedder_calls == 0


async def test_removed_delete_by_external_id_endpoint_returns_not_found(
    asgi_client, in_memory_sqlite
):
    rd = await asgi_client.post(
        "/api/docs/delete_by_external_id",
        json={"external_ids": ["doc-removed"]},
    )
    assert rd.status_code == 404
