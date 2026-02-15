# tests/unit/infrastructure/persistence/sqlalchemy/test_sqlite_document_identity_columns.py

import hashlib

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base
from local_rag_backend.infrastructure.persistence.sqlalchemy.models import Document as DbDocument
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage


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
