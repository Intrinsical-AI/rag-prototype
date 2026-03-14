from __future__ import annotations

from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_import_canonical_upsert_only_and_reimport_is_unchanged(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    payload = {
        "scope": "repogpt:demo",
        "snapshot_id": "snap-1",
        "documents": [
            {
                "external_id": "doc-1",
                "source_id": "file:a.py",
                "content": "def alpha():\n    return 1\n",
                "metadata": {"path": "a.py", "unit_type": "function"},
            }
        ],
    }

    r1 = await asgi_client.post("/api/docs/import-canonical", json=payload)
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["inserted"] == 1
    assert body1["updated"] == 0
    assert body1["unchanged"] == 0
    assert body1["deleted_sql"] == 0

    r2 = await asgi_client.post("/api/docs/import-canonical", json=payload)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["inserted"] == 0
    assert body2["updated"] == 0
    assert body2["unchanged"] == 1

    docs = SqlDocumentStorage().get_all_documents()
    assert len(docs) == 1
    assert docs[0].external_id == "doc-1"
    assert docs[0].metadata == {
        "path": "a.py",
        "unit_type": "function",
        "scope": "repogpt:demo",
        "snapshot_id": "snap-1",
    }


async def test_import_canonical_replace_scope_hard_deletes_stale_docs(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    first = {
        "scope": "repogpt:demo",
        "snapshot_id": "snap-1",
        "replace_scope": True,
        "documents": [
            {
                "external_id": "doc-1",
                "source_id": "file:a.py",
                "content": "def alpha():\n    return 1\n",
                "metadata": {"path": "a.py", "unit_type": "function"},
            },
            {
                "external_id": "doc-2",
                "source_id": "file:b.py",
                "content": "def beta():\n    return 2\n",
                "metadata": {"path": "b.py", "unit_type": "function"},
            },
        ],
    }
    second = {
        "scope": "repogpt:demo",
        "snapshot_id": "snap-2",
        "replace_scope": True,
        "documents": [
            {
                "external_id": "doc-2",
                "source_id": "file:b.py",
                "content": "def beta():\n    return 20\n",
                "metadata": {"path": "b.py", "unit_type": "function"},
            }
        ],
    }

    r1 = await asgi_client.post("/api/docs/import-canonical", json=first)
    assert r1.status_code == 200
    r2 = await asgi_client.post("/api/docs/import-canonical", json=second)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["inserted"] == 0
    assert body2["updated"] == 1
    assert body2["deleted_sql"] == 1
    assert body2["deleted_external_ids"] == ["doc-1"]

    docs = SqlDocumentStorage().get_all_documents()
    assert {doc.external_id for doc in docs} == {"doc-2"}

    third = {
        "scope": "repogpt:demo",
        "snapshot_id": "snap-3",
        "replace_scope": False,
        "documents": [
            {
                "external_id": "doc-1",
                "source_id": "file:a.py",
                "content": "def alpha():\n    return 10\n",
                "metadata": {"path": "a.py", "unit_type": "function"},
            }
        ],
    }
    r3 = await asgi_client.post("/api/docs/import-canonical", json=third)
    assert r3.status_code == 200
    assert r3.json()["inserted"] == 1
    assert {doc.external_id for doc in SqlDocumentStorage().get_all_documents()} == {"doc-1", "doc-2"}


async def test_import_canonical_rejects_duplicate_external_ids(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    payload = {
        "scope": "repogpt:demo",
        "snapshot_id": "snap-1",
        "documents": [
            {"external_id": "doc-1", "content": "alpha"},
            {"external_id": "doc-1", "content": "beta"},
        ],
    }

    response = await asgi_client.post("/api/docs/import-canonical", json=payload)
    assert response.status_code == 422
