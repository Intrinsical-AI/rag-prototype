# tests/test_sql_storage.py
import pytest

from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage, base as db_base


def test_store_and_get_documents(in_memory_sqlite):
    # Use the session factory returned by the fixture
    storage = SqlDocumentStorage(session_factory=in_memory_sqlite)
    texts = ["primer doc", "segundo doc"]

    # --- create ---
    ids = storage.store_documents(texts)
    assert len(ids) == 2
    assert all(isinstance(i, str) and i.startswith("doc:") for i in ids)
    assert len(set(ids)) == 2  # no repetidos

    # --- retrieve ---
    docs = storage.get(ids)
    contents = sorted(d.content for d in docs)
    assert contents == sorted(texts)


def test_delete_by_external_ids_accepts_duplicates_without_integrity_error(in_memory_sqlite):
    storage = SqlDocumentStorage(session_factory=in_memory_sqlite)
    storage.upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id="doc-1", content="hello")]
    )

    deleted_sql, deleted_ids, missing, tombstoned = storage.delete_by_external_ids(
        [" doc-1 ", "doc-1", "doc-1"]
    )
    assert deleted_sql == 1
    assert len(deleted_ids) == 1
    assert missing == []
    assert tombstoned == 1


def test_sql_document_storage_obeys_shared_uow_commit(in_memory_sqlite):
    storage = SqlDocumentStorage(session_factory=in_memory_sqlite)

    with db_base.session_uow(session_factory=in_memory_sqlite):
        ids = storage.store_documents(["inside-uow"])
        docs = storage.get(ids)
        assert len(docs) == 1
        assert docs[0].content == "inside-uow"

    all_docs = storage.get_all_documents()
    assert [d.content for d in all_docs] == ["inside-uow"]


def test_sql_document_storage_obeys_shared_uow_rollback(in_memory_sqlite):
    storage = SqlDocumentStorage(session_factory=in_memory_sqlite)

    with (
        pytest.raises(RuntimeError, match="boom"),
        db_base.session_uow(session_factory=in_memory_sqlite),
    ):
        storage.store_documents(["must-rollback"])
        raise RuntimeError("boom")

    all_docs = storage.get_all_documents()
    assert all_docs == []


def test_snapshot_selectors_use_same_before_image_shape(in_memory_sqlite):
    storage = SqlDocumentStorage(session_factory=in_memory_sqlite)
    results, _, _ = storage.upsert_documents_by_external_id(
        [
            storage.UpsertDoc(
                external_id="entry",
                content="Raw content",
                source_id="source",
                scope="scope",
                snapshot_id="snapshot",
                metadata={"nested": {"value": 1}},
            )
        ]
    )
    plain_id = storage.store_documents(["Plain content"])[0]

    snapshots_by_id = storage.snapshot_by_ids([results[0].id, plain_id])
    snapshots_by_external_id = storage.snapshot_by_external_ids([" entry ", "entry"])

    assert len(snapshots_by_id) == 2
    assert snapshots_by_external_id == [
        next(snap for snap in snapshots_by_id if snap["external_id"] == "entry")
    ]
    assert {snap["external_id"] for snap in snapshots_by_id} == {"entry", None}
    assert set(snapshots_by_external_id[0]) == {
        "id",
        "external_id",
        "content",
        "source_id",
        "scope",
        "snapshot_id",
        "metadata",
        "content_sha256",
    }
    assert snapshots_by_external_id[0]["metadata"] == {"nested": {"value": 1}}
