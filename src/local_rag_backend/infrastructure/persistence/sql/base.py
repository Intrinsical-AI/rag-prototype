# src/infrastructure/persistence/sql/base.py
"""SQLAlchemy engine, session, and base class setup."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.engine import Engine

engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def ensure_sqlite_schema_compatible(
    *,
    engine_to_use: Engine | None = None,
    id_map_path: str | None = None,
) -> None:
    """
    Ensure ORM tables for fresh-install runtime contract.

    Legacy schema migration is intentionally out of scope.
    """
    _ = id_map_path
    eng = engine_to_use or engine
    Base.metadata.create_all(bind=eng)
    ensure_sqlite_documents_autoincrement(engine_to_use=eng, id_map_path=id_map_path)
    ensure_sqlite_documents_identity_columns(engine_to_use=eng)


def ensure_sqlite_documents_autoincrement(
    *, engine_to_use: Engine | None = None, id_map_path: str | None = None
) -> None:
    """Fresh-install contract: AUTOINCREMENT migration is intentionally unsupported."""
    _ = engine_to_use
    _ = id_map_path


def ensure_sqlite_documents_identity_columns(*, engine_to_use: Engine | None = None) -> None:
    """Fresh-install contract: legacy identity-column migration is intentionally unsupported."""
    _ = engine_to_use


async def get_db() -> AsyncGenerator[Session, None]:
    """FastAPI dependency to provide a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
