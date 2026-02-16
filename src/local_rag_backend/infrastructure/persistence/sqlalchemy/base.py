# src/infrastructure/persistence/sqlalchemy/base.py
"""SQLAlchemy engine, session, and base class setup."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.engine import Engine

engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def _is_sqlite_duplicate_column_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "duplicate column name" in msg or "already exists" in msg


def _add_sqlite_column_best_effort(*, conn: Any, ddl_sql: str) -> None:
    try:
        conn.execute(text(ddl_sql))
    except OperationalError as exc:
        # Multi-process startup race: another worker may add the same column between
        # PRAGMA introspection and ALTER TABLE execution.
        if _is_sqlite_duplicate_column_error(exc):
            return
        raise


def ensure_sqlite_documents_autoincrement(
    *, engine_to_use: Engine | None = None, id_map_path: str | None = None
) -> None:
    """
    Ensure the `documents` table uses SQLite AUTOINCREMENT to prevent ID reuse after deletes.

    Why it matters:
    - Without AUTOINCREMENT, SQLite may reuse integer primary keys when the table becomes empty.
    - In dense/hybrid mode (SQL + vector index), ID reuse can map old vectors to new content.

    This function performs a conservative in-place migration if it detects a legacy table
    without AUTOINCREMENT and the schema matches the expected columns (id, content).
    """
    eng = engine_to_use or engine
    if eng.dialect.name != "sqlite":
        return

    with eng.begin() as conn:
        sql = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'")
        ).scalar()
        if not sql:
            return
        if "AUTOINCREMENT" in str(sql).upper():
            return

        cols = conn.execute(text("PRAGMA table_info(documents)")).fetchall()
        col_names = {row[1] for row in cols}  # row[1] = name
        if col_names != {"id", "content"}:
            raise RuntimeError(
                "Legacy SQLite schema detected for table 'documents' (missing AUTOINCREMENT), "
                "but the column set is unexpected. Refusing to auto-migrate."
            )

        # Migrate via copy-into-new-table + swap. IDs are preserved.
        conn.execute(text("DROP TABLE IF EXISTS documents__new"))
        conn.execute(
            text(
                "CREATE TABLE documents__new ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "
                "content TEXT NOT NULL"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO documents__new(id, content) SELECT id, content FROM documents ORDER BY id"
            )
        )
        conn.execute(text("DROP TABLE documents"))
        conn.execute(text("ALTER TABLE documents__new RENAME TO documents"))

        # Seed sqlite_sequence so the next insert does not reuse IDs even if table becomes empty.
        max_id_db = int(
            conn.execute(text("SELECT COALESCE(MAX(id), 0) FROM documents")).scalar() or 0
        )
        max_id_map = 0
        try:
            p = Path(id_map_path or settings.id_map_path)
            if p.is_file():
                loaded = json.loads(p.read_text(encoding="utf-8") or "[]")
                if isinstance(loaded, list) and loaded and all(isinstance(x, int) for x in loaded):
                    max_id_map = int(max(loaded))
        except Exception:
            max_id_map = 0

        max_known = max(max_id_db, max_id_map)
        if max_known > 0:
            seq_exists = conn.execute(
                text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
            ).scalar()
            if not seq_exists:
                # Force SQLite to create sqlite_sequence for AUTOINCREMENT tables.
                # Insert+delete keeps user data unchanged while advancing the sequence.
                res = conn.execute(
                    text("INSERT INTO documents(content) VALUES ('__autoincrement_seed__')")
                )
                seed_id = int(getattr(res, "lastrowid", 0) or 0)
                if seed_id:
                    conn.execute(text("DELETE FROM documents WHERE id=:id"), {"id": seed_id})

            upd = conn.execute(
                text("UPDATE sqlite_sequence SET seq=:seq WHERE name='documents'"),
                {"seq": int(max_known)},
            )
            if getattr(upd, "rowcount", 0) == 0:
                conn.execute(
                    text("INSERT INTO sqlite_sequence(name, seq) VALUES ('documents', :seq)"),
                    {"seq": int(max_known)},
                )


def ensure_sqlite_documents_identity_columns(*, engine_to_use: Engine | None = None) -> None:
    """
    Ensure the `documents` table contains stable identity + metadata columns.

    This is a best-effort, additive migration intended for SQLite deployments without Alembic.
    It never drops data. It:
    - Adds missing columns (external_id, source_id, metadata, content_sha256, created_at, updated_at)
    - Backfills metadata/content_sha256/timestamps for existing rows when possible
    - Creates a unique index for external_id (nullable; multiple NULLs allowed)
    """
    eng = engine_to_use or engine
    if eng.dialect.name != "sqlite":
        return

    required_cols = {
        "external_id",
        "source_id",
        "metadata",
        "content_sha256",
        "chunk_dedup_sha256",
        "created_at",
        "updated_at",
    }

    with eng.begin() as conn:
        sql = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'")
        ).scalar()
        if not sql:
            return

        cols = conn.execute(text("PRAGMA table_info(documents)")).fetchall()
        col_names = {row[1] for row in cols}  # row[1] = name

        missing = required_cols - col_names
        if "external_id" in missing:
            _add_sqlite_column_best_effort(
                conn=conn, ddl_sql="ALTER TABLE documents ADD COLUMN external_id TEXT"
            )
        if "source_id" in missing:
            _add_sqlite_column_best_effort(
                conn=conn, ddl_sql="ALTER TABLE documents ADD COLUMN source_id TEXT"
            )
        if "metadata" in missing:
            _add_sqlite_column_best_effort(
                conn=conn, ddl_sql="ALTER TABLE documents ADD COLUMN metadata TEXT"
            )
        if "content_sha256" in missing:
            _add_sqlite_column_best_effort(
                conn=conn, ddl_sql="ALTER TABLE documents ADD COLUMN content_sha256 TEXT"
            )
        if "chunk_dedup_sha256" in missing:
            _add_sqlite_column_best_effort(
                conn=conn, ddl_sql="ALTER TABLE documents ADD COLUMN chunk_dedup_sha256 TEXT"
            )
        if "created_at" in missing:
            _add_sqlite_column_best_effort(
                conn=conn, ddl_sql="ALTER TABLE documents ADD COLUMN created_at DATETIME"
            )
        if "updated_at" in missing:
            _add_sqlite_column_best_effort(
                conn=conn, ddl_sql="ALTER TABLE documents ADD COLUMN updated_at DATETIME"
            )

        # Indexes (idempotent). Partial index keeps multiple NULLs and enforces uniqueness otherwise.
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_external_id "
                "ON documents(external_id) WHERE external_id IS NOT NULL"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_chunk_dedup_sha256 "
                "ON documents(chunk_dedup_sha256) WHERE chunk_dedup_sha256 IS NOT NULL"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_documents_source_id ON documents(source_id)")
        )

        # Backfills (idempotent)
        conn.execute(text("UPDATE documents SET metadata='{}' WHERE metadata IS NULL"))

        now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        # created_at/updated_at can be NULL for legacy rows: fill them with "now" for consistency.
        conn.execute(
            text("UPDATE documents SET created_at=:now WHERE created_at IS NULL"), {"now": now}
        )
        conn.execute(
            text("UPDATE documents SET updated_at=:now WHERE updated_at IS NULL"), {"now": now}
        )

        # content_sha256: compute in python for rows where missing.
        rows = conn.execute(
            text(
                "SELECT id, content FROM documents "
                "WHERE content_sha256 IS NULL OR content_sha256 = ''"
            )
        ).fetchall()
        for doc_id, content in rows:
            if content is None:
                continue
            h = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
            conn.execute(
                text("UPDATE documents SET content_sha256=:h WHERE id=:id"),
                {"h": h, "id": int(doc_id)},
            )


async def get_db() -> AsyncGenerator[Session, None]:
    """FastAPI dependency to provide a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
