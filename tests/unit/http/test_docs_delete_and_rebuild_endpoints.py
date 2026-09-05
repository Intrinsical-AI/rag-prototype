import pytest

from local_rag_backend.settings import settings


@pytest.mark.unit
async def test_mutate_delete_ids_sparse_removes_from_sql(
    asgi_client, in_memory_sqlite, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    r = await asgi_client.post("/api/docs/ingest", json={"texts": ["a", "b", "c"]})
    assert r.status_code == 200
    ids = r.json()["ids"]

    r2 = await asgi_client.post("/api/docs/mutate", json={"delete_ids": [ids[0], ids[2]]})
    assert r2.status_code == 200
    assert r2.json()["deleted_sql"] == 2

    r3 = await asgi_client.post("/api/docs/query", json={"limit": 100, "offset": 0, "filters": []})
    assert r3.status_code == 200
    remaining_ids = {d["id"] for d in r3.json()}
    assert ids[0] not in remaining_ids
    assert ids[2] not in remaining_ids


@pytest.mark.unit
async def test_rebuild_index_dense_from_db(tmp_path, asgi_client, in_memory_sqlite, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "idx.npy"), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(tmp_path / "id_map.json"), raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    class DummyEmbedder:
        dim = 2

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(t)), 0.0] for t in texts]

    monkeypatch.setattr(
        "local_rag_backend.composition.factory.SentenceTransformerEmbedder",
        lambda model_name=None: DummyEmbedder(),
        raising=True,
    )

    r = await asgi_client.post("/api/docs/ingest", json={"texts": ["alpha", "beta"]})
    assert r.status_code == 200
    ids = r.json()["ids"]

    rr = await asgi_client.post("/api/index/rebuild", json={})
    assert rr.status_code == 200
    assert rr.json()["indexed"] == 2

    rd = await asgi_client.post("/api/docs/mutate", json={"delete_ids": [ids[0]]})
    assert rd.status_code == 200

    rr3 = await asgi_client.post("/api/index/rebuild", json={})
    assert rr3.status_code == 200
    assert rr3.json()["indexed"] == 1


@pytest.mark.unit
async def test_mutate_delete_dense_does_not_require_embedder_when_no_upserts(
    asgi_client, in_memory_sqlite, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    r = await asgi_client.post("/api/docs/ingest", json={"texts": ["zeta"]})
    assert r.status_code == 200
    doc_id = r.json()["ids"][0]

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    embedder_calls = 0

    def _boom_embedder(**_kwargs: object) -> object:
        nonlocal embedder_calls
        embedder_calls += 1
        raise RuntimeError("embedder should not be called on successful delete path")

    class DummyVec:
        def apply_delta_atomic(self, *, delete_ids: object, upserts: object) -> None:
            assert len(list(delete_ids)) == 1
            assert list(upserts) == []

    monkeypatch.setattr(
        "local_rag_backend.composition.factory.SentenceTransformerEmbedder",
        _boom_embedder,
        raising=True,
    )
    monkeypatch.setattr(
        "local_rag_backend.composition.factory.VectorStorage",
        lambda **_k: DummyVec(),
        raising=True,
    )

    rd = await asgi_client.post("/api/docs/mutate", json={"delete_ids": [doc_id]})
    assert rd.status_code == 200
    payload = rd.json()
    assert payload["deleted_sql"] == 1
    assert payload["deleted_index"] == 1
    assert payload["index_rebuilt"] is False
    assert embedder_calls == 0


@pytest.mark.unit
async def test_removed_delete_endpoint_returns_not_found(asgi_client, in_memory_sqlite) -> None:
    r = await asgi_client.post(
        "/api/docs/delete",
        json={"ids": ["doc-removed"]},
    )
    assert r.status_code == 404
