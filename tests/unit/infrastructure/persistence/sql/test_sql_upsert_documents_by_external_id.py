# tests/unit/infrastructure/persistence/sql/test_sql_upsert_documents_by_external_id.py

import pytest

from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage


def test_upsert_inserts_and_is_idempotent(in_memory_sqlite):
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    items = [
        SqlDocumentStorage.UpsertDoc(
            external_id="doc-1",
            content="hello",
            source_id="src-a",
            metadata={"k": "v"},
        )
    ]

    res1, changed1, updated_ids1 = repo.upsert_documents_by_external_id(items)
    assert len(res1) == 1
    assert res1[0].action == "inserted"
    assert res1[0].content_changed is True
    assert changed1 and changed1[0][0] == res1[0].id
    assert updated_ids1 == []

    # Idempotent: same payload does not create a new row and does not require re-embedding.
    res2, changed2, updated_ids2 = repo.upsert_documents_by_external_id(items)
    assert len(res2) == 1
    assert res2[0].id == res1[0].id
    assert res2[0].action == "unchanged"
    assert changed2 == []
    assert updated_ids2 == []

    doc = repo.get([res1[0].id])[0]
    assert doc.external_id == "doc-1"
    assert doc.source_id == "src-a"
    assert doc.metadata == {"k": "v"}


def test_upsert_updates_content_and_hash(in_memory_sqlite):
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    first, *_ = repo.upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id="doc-1", content="hello")]
    )
    doc_id = first[0].id

    res2, changed2, updated_ids2 = repo.upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id="doc-1", content="HELLO2")]
    )
    assert res2[0].id == doc_id
    assert res2[0].action == "updated"
    assert res2[0].content_changed is True
    assert changed2 == [(doc_id, "HELLO2")]
    assert updated_ids2 == [doc_id]

    doc = repo.get([doc_id])[0]
    assert doc.content == "HELLO2"


def test_upsert_metadata_only_change(in_memory_sqlite):
    """Metadata update triggers action=updated with content_changed=False."""
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    first, *_ = repo.upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id="doc-1", content="same", metadata={"v": 1})]
    )
    doc_id = first[0].id

    res, changed, updated_ids = repo.upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id="doc-1", content="same", metadata={"v": 2})]
    )
    assert res[0].action == "updated"
    assert res[0].content_changed is False
    assert changed == []
    assert updated_ids == []
    assert repo.get([doc_id])[0].metadata == {"v": 2}


def test_upsert_source_only_change(in_memory_sqlite):
    """Source update triggers action=updated with content_changed=False."""
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    first, *_ = repo.upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id="doc-1", content="text", source_id="src-a")]
    )
    doc_id = first[0].id

    res, changed, updated_ids = repo.upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id="doc-1", content="text", source_id="src-b")]
    )
    assert res[0].action == "updated"
    assert res[0].content_changed is False
    assert changed == []
    assert updated_ids == []
    assert repo.get([doc_id])[0].source_id == "src-b"


def test_upsert_dedup_only_change(in_memory_sqlite):
    """chunk_dedup_sha256 update triggers action=updated with content_changed=False."""
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    first, *_ = repo.upsert_documents_by_external_id(
        [
            SqlDocumentStorage.UpsertDoc(
                external_id="doc-1", content="text", chunk_dedup_sha256="sha-v1"
            )
        ]
    )
    doc_id = first[0].id

    res, changed, updated_ids = repo.upsert_documents_by_external_id(
        [
            SqlDocumentStorage.UpsertDoc(
                external_id="doc-1", content="text", chunk_dedup_sha256="sha-v2"
            )
        ]
    )
    assert res[0].action == "updated"
    assert res[0].content_changed is False
    assert changed == []
    assert updated_ids == []


def test_upsert_rejects_duplicate_external_ids(in_memory_sqlite):
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    with pytest.raises(ValueError, match="unique within the request"):
        repo.upsert_documents_by_external_id(
            [
                SqlDocumentStorage.UpsertDoc(external_id="doc-1", content="a"),
                SqlDocumentStorage.UpsertDoc(external_id="doc-1", content="b"),
            ]
        )
