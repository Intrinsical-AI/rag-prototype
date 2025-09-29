# src/infrastructure/persistence/sqlalchemy/sql_.py
"""
SQLAlchemy-based implementation of the document and history repositories.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from local_rag_backend.core.domain.entities import Document as DomainDocument
from local_rag_backend.core.ports import DocumentRepoPort, QAHistoryPort
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import (
    SessionLocal,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy.crud import (
    add_documents,
    add_history,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy.models import Document as DbDocument

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from sqlalchemy.orm import Session, sessionmaker


@contextmanager
def get_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


class SqlDocumentStorage(DocumentRepoPort):
    """SQL-based implementation of the document repository port."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._session_factory = session_factory or SessionLocal

    def store_documents(self, texts: Sequence[str]) -> list[int]:
        """Store documents in the database."""
        with get_session(self._session_factory) as session:
            return add_documents(session, list(texts))

    def get(self, ids: Sequence[int]) -> Sequence[DomainDocument]:
        """Retrieve documents by their IDs."""
        with get_session(self._session_factory) as session:
            db_docs = session.query(DbDocument).filter(DbDocument.id.in_(ids)).all()
            return [DomainDocument(id=d.id, content=d.content) for d in db_docs]

    def get_all_documents(self) -> Sequence[DomainDocument]:
        """Retrieve all documents from the database."""
        with get_session(self._session_factory) as session:
            db_docs = session.query(DbDocument).order_by(DbDocument.id).all()
            return [DomainDocument(id=d.id, content=d.content) for d in db_docs]


class HistorySqlStorage(QAHistoryPort):
    """SQL-based implementation of the history repository port."""

    def save(self, q: str, a: str, source_ids: Sequence[int]) -> None:
        """Save a question-answer pair to the history table."""
        with get_session(SessionLocal) as session:
            add_history(session, q, a, source_ids=list(source_ids))
