import pytest

from local_rag_backend.settings import settings


@pytest.mark.unit
async def test_delete_docs_sparse_removes_from_sql(asgi_client, in_memory_sqlite, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    r = await asgi_client.post("/api/docs", json={"texts": ["a", "b", "c"]})
    assert r.status_code == 200
    ids = r.json()["ids"]

    r2 = await asgi_client.post("/api/docs/delete", json={"ids": [ids[0], ids[2]]})
    assert r2.status_code == 200
    assert r2.json()["deleted_sql"] == 2

    r3 = await asgi_client.get("/api/docs", params={"limit": 100, "offset": 0})
    assert r3.status_code == 200
    remaining_ids = {d["id"] for d in r3.json()}
    assert ids[0] not in remaining_ids
    assert ids[2] not in remaining_ids


@pytest.mark.unit
async def test_rebuild_index_dense_from_db(tmp_path, asgi_client, in_memory_sqlite, monkeypatch):
    # Dense mode with numpy fallback (no faiss required).
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "idx.npy"), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(tmp_path / "id_map.json"), raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    # Dummy embedder to avoid heavy deps.
    class DummyEmbedder:
        dim = 2

        def embed(self, texts):
            return [[float(len(t)), 0.0] for t in texts]

    monkeypatch.setattr(
        "local_rag_backend.app.api_router.SentenceTransformerEmbedder",
        lambda model_name=None: DummyEmbedder(),
        raising=True,
    )

    r = await asgi_client.post("/api/docs", json={"texts": ["alpha", "beta"]})
    assert r.status_code == 200
    ids = r.json()["ids"]

    # First rebuild
    rr = await asgi_client.post("/api/index/rebuild", json={})
    assert rr.status_code == 200
    assert rr.json()["indexed"] == 2

    # Second rebuild should be idempotent.
    rr2 = await asgi_client.post("/api/index/rebuild", json={})
    assert rr2.status_code == 200
    assert rr2.json()["indexed"] == 2

    # Delete one doc and rebuild; index should contain only remaining ID.
    rd = await asgi_client.post("/api/docs/delete", json={"ids": [ids[0]]})
    assert rd.status_code == 200

    rr3 = await asgi_client.post("/api/index/rebuild", json={})
    assert rr3.status_code == 200
    assert rr3.json()["indexed"] == 1
