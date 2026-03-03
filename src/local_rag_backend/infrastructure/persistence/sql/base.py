# src/infrastructure/persistence/sql/base.py
"""SQLAlchemy engine, session, and base class setup."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from sqlalchemy.engine import Engine

engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()
_BOUND_SESSION: ContextVar[Session | None] = ContextVar("_BOUND_SESSION", default=None)


def get_bound_session() -> Session | None:
    """Return the current SQL session bound to the active unit-of-work context, if any."""
    return _BOUND_SESSION.get()


@contextmanager
def session_uow(
    *, session_factory: sessionmaker[Session] | None = None
) -> Generator[None, None, None]:
    """Run a SQL unit-of-work and bind its session to the current context."""
    factory = session_factory or SessionLocal
    with factory.begin() as session:
        token = _BOUND_SESSION.set(session)
        try:
            yield
        finally:
            _BOUND_SESSION.reset(token)


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


async def get_db() -> AsyncGenerator[Session, None]:
    """FastAPI dependency to provide a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
