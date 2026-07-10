import numpy as np

from local_rag_backend.composition import factory
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_mutate_upsert_sparse_is_idempotent(asgi_client, in_memory_sqlite, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    payload = {"upserts": [{"external_id": "doc-1", "content": "hello"}]}
    r1 = await asgi_client.post("/api/docs/mutate", json=payload)
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["inserted"] == 1
    assert data1["updated"] == 0
    assert data1["unchanged"] == 0
    doc_id = data1["results"][0]["id"]

    r2 = await asgi_client.post("/api/docs/mutate", json=payload)
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["inserted"] == 0
    assert data2["updated"] == 0
    assert data2["unchanged"] == 1
    assert data2["results"][0]["id"] == doc_id

    r3 = await asgi_client.post(
        "/api/docs/mutate",
        json={"upserts": [{"external_id": "doc-1", "content": "hello2"}]},
    )
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3["inserted"] == 0
    assert data3["updated"] == 1
    assert data3["unchanged"] == 0
    assert data3["results"][0]["id"] == doc_id

    doc = SqlDocumentStorage().get([doc_id])[0]
    assert doc.content == "hello2"


async def test_mutate_upsert_rejects_duplicate_external_ids(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    payload = {
        "upserts": [
            {"external_id": "doc-1", "content": "a"},
            {"external_id": "doc-1", "content": "b"},
        ]
    }
    r = await asgi_client.post("/api/docs/mutate", json=payload)
    assert r.status_code == 400


async def test_mutate_upsert_dense_updates_only_changed_content(
    asgi_client, in_memory_sqlite, monkeypatch, tmp_path
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
            self.calls: list[dict[str, list[str]]] = []

        def apply_delta_atomic(self, *, delete_ids, upserts):
            self.calls.append(
                {
                    "delete_ids": [str(x) for x in delete_ids],
                    "upsert_ids": [str(doc_id) for doc_id, _ in upserts],
                }
            )

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(
        settings,
        "embedding_cache_db_path",
        str(tmp_path / "embedding-cache.sqlite3"),
        raising=False,
    )

    dummy_embedder = DummyEmbedder()
    dummy_vec = DummyVec()

    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: dummy_embedder)
    monkeypatch.setattr(factory, "VectorStorage", lambda *a, **k: dummy_vec)

    payload = {"upserts": [{"external_id": "doc-1", "content": "hello"}]}
    r1 = await asgi_client.post("/api/docs/mutate", json=payload)
    assert r1.status_code == 200
    doc_id = r1.json()["results"][0]["id"]
    assert dummy_embedder.calls == 1
    assert len(dummy_vec.calls) == 1
    assert dummy_vec.calls[0]["delete_ids"] == []
    assert dummy_vec.calls[0]["upsert_ids"] == [doc_id]

    r2 = await asgi_client.post("/api/docs/mutate", json=payload)
    assert r2.status_code == 200
    # The dense embedder is wrapped by the persistent content-addressed cache, so
    # unchanged content should not hit the underlying provider twice.
    assert dummy_embedder.calls == 1
    assert len(dummy_vec.calls) == 2
    assert dummy_vec.calls[1]["delete_ids"] == []
    assert dummy_vec.calls[1]["upsert_ids"] == []

    r3 = await asgi_client.post(
        "/api/docs/mutate",
        json={"upserts": [{"external_id": "doc-1", "content": "hello2"}]},
    )
    assert r3.status_code == 200
    assert r3.json()["results"][0]["id"] == doc_id
    assert dummy_embedder.calls == 2
    assert len(dummy_vec.calls) == 3
    assert dummy_vec.calls[2]["delete_ids"] == [doc_id]
    assert dummy_vec.calls[2]["upsert_ids"] == [doc_id]


async def test_legacy_upsert_endpoint_is_removed(asgi_client, in_memory_sqlite):
    r = await asgi_client.post(
        "/api/docs/upsert",
        json={"docs": [{"external_id": "doc-legacy", "content": "x"}]},
    )
    assert r.status_code == 404
