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


@pytest.mark.unit
async def test_delete_docs_dense_does_not_require_embedder_when_index_delete_succeeds(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    r = await asgi_client.post("/api/docs", json={"texts": ["zeta"]})
    assert r.status_code == 200
    doc_id = r.json()["ids"][0]

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    embedder_calls = 0

    def _boom_embedder(**_kwargs):
        nonlocal embedder_calls
        embedder_calls += 1
        raise RuntimeError("embedder should not be called on successful delete path")

    class DummyVec:
        def delete(self, ids):
            return len(list(ids))

    monkeypatch.setattr(
        "local_rag_backend.app.api_router.SentenceTransformerEmbedder",
        _boom_embedder,
        raising=True,
    )
    monkeypatch.setattr(
        "local_rag_backend.app.api_router.FaissVectorStorage",
        lambda **_k: DummyVec(),
        raising=True,
    )

    rd = await asgi_client.post("/api/docs/delete", json={"ids": [doc_id]})
    assert rd.status_code == 200
    payload = rd.json()
    assert payload["deleted_sql"] == 1
    assert payload["deleted_index"] == 1
    assert payload["rebuilt_index"] is False
    assert embedder_calls == 0
