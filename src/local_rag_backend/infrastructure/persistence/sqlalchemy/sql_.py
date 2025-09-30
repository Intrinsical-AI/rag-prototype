"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

SQLAlchemy-based implementation of the document and history repositories.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from local_rag_backend.core.domain.entities import Document as DomainDocument
from local_rag_backend.core.ports import DocumentRepoPort, QAHistoryPort
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import SessionLocal
from local_rag_backend.infrastructure.persistence.sqlalchemy.crud import add_documents, add_history
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
        """Retrieve documents by their IDs in the same order as requested.
        
        Args:
            ids: Sequence of document IDs to retrieve
            
        Returns:
            Sequence of documents in the same order as input IDs.
            Missing documents are skipped (not included in result).
            
        Note:
            Preserves the order of input IDs, which is critical for maintaining
            correspondence with scores in retrieval operations.
        """
        if not ids:
            return []
            
        with get_session(self._session_factory) as session:
            # Query all documents at once for efficiency
            db_docs = session.query(DbDocument).filter(DbDocument.id.in_(ids)).all()
            
            # Create a mapping for O(1) lookup
            docs_by_id = {doc.id: doc for doc in db_docs}
            
            # Return documents in the same order as input IDs, skipping missing ones
            ordered_docs = []
            for doc_id in ids:
                if doc_id in docs_by_id:
                    db_doc = docs_by_id[doc_id]
                    ordered_docs.append(DomainDocument(id=db_doc.id, content=db_doc.content))
            
            return ordered_docs

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
