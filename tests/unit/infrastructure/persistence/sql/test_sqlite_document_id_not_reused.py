from local_rag_backend.infrastructure.persistence.sql import base as db_base
from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage


def test_document_ids_are_not_reused_after_delete(in_memory_sqlite):
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    first_ids = list(repo.store_documents(["a"]))
    assert len(first_ids) == 1

    repo.delete_documents(first_ids)

    second_ids = list(repo.store_documents(["b"]))
    assert len(second_ids) == 1

    assert second_ids[0] != first_ids[0]
    assert str(first_ids[0]).startswith("doc:")
    assert str(second_ids[0]).startswith("doc:")


def test_noop_schema_helpers_are_available():
    # Fresh-install only contract: helpers are intentionally no-op, but callable.
    db_base.ensure_sqlite_documents_autoincrement()
    db_base.ensure_sqlite_documents_identity_columns()
