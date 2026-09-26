# src/infrastructure/persistence/sql/base.py
"""SQLAlchemy engine, session, and base class setup."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session, declarative_base, sessionmaker

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy.engine import Engine

Base = declarative_base()
_BOUND_SESSION: ContextVar[tuple[sessionmaker[Session], Session] | None] = ContextVar(
    "_BOUND_SESSION", default=None
)


def get_bound_session(session_factory: sessionmaker[Session]) -> Session | None:
    """Return the current SQL session bound to the active unit-of-work context, if any."""
    bound = _BOUND_SESSION.get()
    return bound[1] if bound is not None and bound[0] is session_factory else None


@contextmanager
def session_uow(*, session_factory: sessionmaker[Session]) -> Generator[None, None, None]:
    """Run a SQL unit-of-work and bind its session to the current context."""
    with session_factory.begin() as session:
        token = _BOUND_SESSION.set((session_factory, session))
        try:
            yield
        finally:
            _BOUND_SESSION.reset(token)


def ensure_sqlite_schema_current(
    *,
    engine_to_use: Engine,
) -> None:
    """
    Ensure ORM tables for fresh-install runtime contract.

    Schema migration is intentionally out of scope.
    """
    Base.metadata.create_all(bind=engine_to_use)
