# tests/test_sql_storage.py
from src.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage


def test_store_and_get_documents(in_memory_sqlite):
    storage = SqlDocumentStorage()
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
