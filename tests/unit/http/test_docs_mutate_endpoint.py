from __future__ import annotations

from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_docs_mutate_sparse_upsert_and_delete_by_external_id(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    r1 = await asgi_client.post(
        "/api/docs/mutate",
        json={
            "op_id": "op-upsert-1",
            "upserts": [{"external_id": "doc-1", "content": "hello"}],
        },
    )
    assert r1.status_code == 200
    p1 = r1.json()
    assert p1["op_id"] == "op-upsert-1"
    assert p1["inserted"] == 1
    assert p1["updated"] == 0
    assert p1["unchanged"] == 0
    assert len(p1["results"]) == 1

    # Repeat same mutation idempotently; should not duplicate SQL rows.
    r2 = await asgi_client.post(
        "/api/docs/mutate",
        json={
            "op_id": "op-upsert-1",
            "upserts": [{"external_id": "doc-1", "content": "hello"}],
        },
    )
    assert r2.status_code == 200
    p2 = r2.json()
    assert p2["inserted"] == 0
    assert p2["updated"] == 0
    assert p2["unchanged"] == 1

    docs = SqlDocumentStorage().get_all_documents()
    assert len(docs) == 1

    r3 = await asgi_client.post(
        "/api/docs/mutate",
        json={
            "op_id": "op-del-1",
            "delete_external_ids": ["doc-1"],
        },
    )
    assert r3.status_code == 200
    p3 = r3.json()
    assert p3["deleted_sql"] == 1
    assert p3["tombstoned"] == 1

    assert SqlDocumentStorage().get_all_documents() == []


async def test_docs_mutate_rejects_conflicting_upsert_and_delete_external_id(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    resp = await asgi_client.post(
        "/api/docs/mutate",
        json={
            "upserts": [{"external_id": "doc-1", "content": "hello"}],
            "delete_external_ids": ["doc-1"],
        },
    )

    assert resp.status_code == 422


async def test_docs_mutate_rejects_reusing_op_id_with_different_intent(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    first = await asgi_client.post(
        "/api/docs/mutate",
        json={
            "op_id": "op-replay-1",
            "upserts": [{"external_id": "doc-1", "content": "hello"}],
        },
    )
    assert first.status_code == 200

    second = await asgi_client.post(
        "/api/docs/mutate",
        json={
            "op_id": "op-replay-1",
            "upserts": [{"external_id": "doc-2", "content": "different"}],
        },
    )
    assert second.status_code == 400
    assert "replay mismatch" in second.json()["detail"].lower()
