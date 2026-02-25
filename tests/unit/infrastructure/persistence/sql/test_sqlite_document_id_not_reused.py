import pytest
from sqlalchemy import create_engine, text

from local_rag_backend.infrastructure.persistence.sql import base as db_base
from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage


def test_sqlite_document_ids_are_not_reused_after_delete(in_memory_sqlite):
    """
    Regression test for a critical integrity issue:
    SQLite can reuse INTEGER PRIMARY KEY values after deletes unless AUTOINCREMENT is used.
    In a multi-store system (SQL + FAISS), ID reuse can make old vectors map to new content.
    """
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    first_ids = list(repo.store_documents(["a"]))
    assert len(first_ids) == 1

    repo.delete_documents(first_ids)

    second_ids = list(repo.store_documents(["b"]))
    assert len(second_ids) == 1

    assert second_ids[0] != first_ids[0]


def test_ensure_autoincrement_migrates_legacy_table(tmp_path):
    db_path = tmp_path / "app.db"
    id_map_path = tmp_path / "id_map.json"
    # Simulate legacy vector store referencing a prior ID.
    id_map_path.write_text("[1]", encoding="utf-8")

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        # Legacy schema: missing AUTOINCREMENT.
        conn.execute(
            text("CREATE TABLE documents (id INTEGER PRIMARY KEY NOT NULL, content TEXT NOT NULL)")
        )
        conn.execute(text("INSERT INTO documents(content) VALUES ('a')"))
        conn.execute(text("DELETE FROM documents"))

    db_base.ensure_sqlite_documents_autoincrement(
        engine_to_use=engine, id_map_path=str(id_map_path)
    )

    with engine.begin() as conn:
        conn.execute(text("INSERT INTO documents(content) VALUES ('b')"))
        new_id = int(conn.execute(text("SELECT id FROM documents")).scalar() or 0)

    assert new_id == 2


def test_ensure_autoincrement_noop_when_documents_missing(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'app.db'}", connect_args={"check_same_thread": False}
    )
    # No tables created: should no-op.
    db_base.ensure_sqlite_documents_autoincrement(
        engine_to_use=engine, id_map_path=str(tmp_path / "id_map.json")
    )


def test_ensure_autoincrement_refuses_unknown_schema(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'app.db'}", connect_args={"check_same_thread": False}
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE documents (id INTEGER PRIMARY KEY NOT NULL, content TEXT NOT NULL, extra TEXT)"
            )
        )
    with pytest.raises(RuntimeError, match="Refusing to auto-migrate"):
        db_base.ensure_sqlite_documents_autoincrement(engine_to_use=engine)


def test_ensure_autoincrement_noop_when_not_sqlite():
    class _Dialect:
        name = "postgresql"

    class _Engine:
        dialect = _Dialect()

    # Should return before touching `.begin()`.
    db_base.ensure_sqlite_documents_autoincrement(engine_to_use=_Engine())  # type: ignore[arg-type]
