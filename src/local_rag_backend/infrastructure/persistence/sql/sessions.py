"""Shared SQL session helpers for repository adapters."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING

from local_rag_backend.infrastructure.persistence.sql import base as db_base

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from sqlalchemy.orm import Session, sessionmaker


def _normalize_external_ids(external_ids: Sequence[str]) -> list[str]:
    """
    Normalize external IDs for delete/tombstone operations.

    - Strip whitespace
    - Drop blanks
    - Deduplicate while preserving order
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in external_ids:
        ext = str(raw).strip()
        if not ext or ext in seen:
            continue
        seen.add(ext)
        normalized.append(ext)
    return normalized


@contextmanager
def get_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = session_factory()
    try:
        yield session
    except Exception:  # rollback on any failure — broad guard for non-SQLAlchemy errors too
        with suppress(Exception):
            session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_managed_session(
    session_factory: sessionmaker[Session],
) -> Generator[tuple[Session, bool], None, None]:
    """Yield (session, owns_session) honoring an active SQL unit-of-work if present."""
    bound = db_base.get_bound_session()
    if bound is not None:
        yield bound, False
        return
    with get_session(session_factory) as session:
        yield session, True


__all__ = ["get_managed_session", "get_session"]
