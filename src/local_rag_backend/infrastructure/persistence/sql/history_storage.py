"""History repository adapter backed by SQLAlchemy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.ports import QAHistoryPort
from local_rag_backend.infrastructure.persistence.sql import base as db_base
from local_rag_backend.infrastructure.persistence.sql.crud import add_history
from local_rag_backend.infrastructure.persistence.sql.sessions import get_managed_session

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session, sessionmaker


class HistorySqlStorage(QAHistoryPort):
    """SQL-based implementation of the history repository port."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._session_factory = session_factory or db_base.SessionLocal

    def save(self, q: str, a: str, source_ids: Sequence[DocId]) -> None:
        """Save a question-answer pair to the history table."""
        with get_managed_session(self._session_factory) as (session, owns_session):
            add_history(session, q, a, source_ids=list(source_ids), autocommit=owns_session)


__all__ = ["HistorySqlStorage"]
