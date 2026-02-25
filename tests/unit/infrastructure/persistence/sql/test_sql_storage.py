# tests/test_sql_storage.py
from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage


def test_store_and_get_documents(in_memory_sqlite):
    # Use the session factory returned by the fixture
    storage = SqlDocumentStorage(session_factory=in_memory_sqlite)
    texts = ["primer doc", "segundo doc"]

    # --- create ---
    ids = storage.store_documents(texts)
    assert len(ids) == 2
    assert all(isinstance(i, int) and i > 0 for i in ids)
    assert len(set(ids)) == 2  # no repetidos

    # --- retrieve ---
    docs = storage.get(ids)
    contents = sorted(d.content for d in docs)
    assert contents == sorted(texts)


def test_tombstone_external_ids_deduplicates_request(in_memory_sqlite):
    storage = SqlDocumentStorage(session_factory=in_memory_sqlite)

    added = storage.tombstone_external_ids([" ext-1 ", "ext-1", "ext-1"])
    assert added == 1

    # Idempotent on repeated calls with duplicates.
    added_again = storage.tombstone_external_ids(["ext-1", " ext-1 "])
    assert added_again == 0


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
