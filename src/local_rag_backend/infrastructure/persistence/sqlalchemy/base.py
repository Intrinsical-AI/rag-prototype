# src/infrastructure/persistence/sqlalchemy/base.py
"""SQLAlchemy engine, session, and base class setup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.engine import Engine

engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


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


async def get_db() -> AsyncGenerator[Session, None]:
    """FastAPI dependency to provide a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
