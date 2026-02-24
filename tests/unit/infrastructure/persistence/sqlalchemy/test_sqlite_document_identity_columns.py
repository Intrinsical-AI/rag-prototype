# tests/unit/infrastructure/persistence/sqlalchemy/test_sqlite_document_identity_columns.py

import hashlib

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import sessionmaker

from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base
from local_rag_backend.infrastructure.persistence.sqlalchemy.models import Document as DbDocument
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage


def test_ensure_sqlite_schema_compatible_runs_bootstrap_steps(monkeypatch):
    calls: list[tuple[object, ...]] = []
    dummy_engine = object()

    def _create_all(*, bind):
        calls.append(("create_all", bind))

    def _ensure_autoincrement(*, engine_to_use=None, id_map_path=None):
        calls.append(("autoincrement", engine_to_use, id_map_path))

    def _ensure_identity(*, engine_to_use=None):
        calls.append(("identity_columns", engine_to_use))

    monkeypatch.setattr(db_base.Base.metadata, "create_all", _create_all, raising=False)
    monkeypatch.setattr(
        db_base, "ensure_sqlite_documents_autoincrement", _ensure_autoincrement, raising=True
    )
    monkeypatch.setattr(
        db_base, "ensure_sqlite_documents_identity_columns", _ensure_identity, raising=True
    )

    db_base.ensure_sqlite_schema_compatible(
        engine_to_use=dummy_engine,
        id_map_path="id_map.json",
    )
    assert calls == [
        ("create_all", dummy_engine),
        ("autoincrement", dummy_engine, "id_map.json"),
        ("identity_columns", dummy_engine),
    ]


def test_ensure_identity_columns_adds_and_backfills(tmp_path):
    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    with engine.begin() as conn:
        # Legacy minimal schema (no identity columns).
        conn.execute(
            text("CREATE TABLE documents (id INTEGER PRIMARY KEY NOT NULL, content TEXT NOT NULL)")
        )
        conn.execute(text("INSERT INTO documents(id, content) VALUES (1, 'hello')"))

    # Startup order mirrors app lifespan.
    db_base.ensure_sqlite_documents_autoincrement(engine_to_use=engine)
    db_base.ensure_sqlite_documents_identity_columns(engine_to_use=engine)

    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(documents)")).fetchall()}
        assert {
            "external_id",
            "source_id",
            "metadata",
            "content_sha256",
            "created_at",
            "updated_at",
        }.issubset(cols)

        row = conn.execute(
            text(
                "SELECT external_id, source_id, metadata, content_sha256, created_at, updated_at "
                "FROM documents WHERE id=1"
            )
        ).fetchone()
        assert row is not None
        assert row[0] is None  # external_id
        assert row[1] is None  # source_id
        assert row[2] == "{}"  # metadata backfill (JSON text)
        assert row[3] == hashlib.sha256(b"hello").hexdigest()
        assert row[4] is not None
        assert row[5] is not None


def test_ensure_identity_columns_creates_unique_index_for_external_id(tmp_path):
    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # Create current schema via ORM and then ensure indexes exist (idempotent).
    db_base.Base.metadata.create_all(bind=engine)
    db_base.ensure_sqlite_documents_identity_columns(engine_to_use=engine)

    with SessionLocal() as s:
        s.add(DbDocument(content="a", external_id="X"))
        s.commit()

        s.add(DbDocument(content="b", external_id="X"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_store_documents_sets_content_sha256(in_memory_sqlite):
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    ids = list(repo.store_documents(["a", "b"]))
    docs = repo.get(ids)
    by_content = {d.content: d for d in docs}
    assert by_content["a"].content_sha256 == hashlib.sha256(b"a").hexdigest()
    assert by_content["b"].content_sha256 == hashlib.sha256(b"b").hexdigest()


def test_ensure_identity_columns_tolerates_duplicate_column_race(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE documents (id INTEGER PRIMARY KEY NOT NULL, content TEXT NOT NULL)")
        )

    original_execute = Connection.execute
    injected = False

    def _execute_with_race(self, statement, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal injected
        sql = str(statement)
        if not injected and "ALTER TABLE documents ADD COLUMN source_id TEXT" in sql:
            injected = True
            # Simulate a parallel worker adding the same column between introspection and ALTER.
            with engine.begin() as other:
                other.execute(text("ALTER TABLE documents ADD COLUMN source_id TEXT"))
            raise OperationalError(sql, {}, Exception("duplicate column name: source_id"))
        return original_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(Connection, "execute", _execute_with_race, raising=True)

    db_base.ensure_sqlite_documents_identity_columns(engine_to_use=engine)
    assert injected is True

    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(documents)")).fetchall()}
        assert "source_id" in cols
        assert "external_id" in cols
