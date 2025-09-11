# src/adapters/storage/sql_crud.py

from collections.abc import Sequence

from sqlalchemy.orm import Session, sessionmaker

from local_rag_backend.core.domain.entities import Document as DomainDocument
from local_rag_backend.core.ports import DocumentRepoPort, QAHistoryPort
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import (
    Base,
    SessionLocal,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy.crud import (
    add_documents,
    save_qa_history,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy.models import Document as DbDocument

# src/infrastructure/persistence/sqlalchemy/sql_.py


def _new_session_factory_from_settings() -> sessionmaker[Session]:
    """
    Use the global SessionLocal from base.py to avoid duplicate engines.
    """
    from local_rag_backend.infrastructure.persistence.sqlalchemy.base import SessionLocal

    return SessionLocal


class SqlDocumentStorage(DocumentRepoPort):
    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        # If no factory is provided, create one dynamically from current settings
        self._session_factory = (
            session_factory if session_factory is not None else _new_session_factory_from_settings()
        )

    def store_documents(self, texts: Sequence[str]) -> list[int]:
        session = self._session_factory()
        try:
            # Ensure metadata is created on this connection (idempotent)
            Base.metadata.create_all(bind=session.get_bind())
            return add_documents(session, list(texts))
        finally:
            # Close session; DO NOT dispose engine here (breaks in-memory SQLite and pooling)
            session.close()

    def get(self, ids: Sequence[int]) -> Sequence[DomainDocument]:
        session = self._session_factory()
        try:
            # Ensure metadata is created on this connection (idempotent)
            Base.metadata.create_all(bind=session.get_bind())
            db_docs = session.query(DbDocument).filter(DbDocument.id.in_(ids)).all()
            return [DomainDocument(id=d.id, content=d.content) for d in db_docs]
        finally:
            session.close()

    def get_all_documents(self) -> Sequence[DomainDocument]:
        session = self._session_factory()
        try:
            # Ensure metadata is created on this connection (idempotent)
            Base.metadata.create_all(bind=session.get_bind())
            db_docs = session.query(DbDocument).order_by(DbDocument.id).all()
            return [DomainDocument(id=d.id, content=d.content) for d in db_docs]
        finally:
            session.close()

    save = store_documents


class HistorySqlStorage(QAHistoryPort):
    def save(self, q: str, a: str, source_ids: Sequence[int]) -> None:
        session = SessionLocal()
        try:
            save_qa_history(session, q, a, source_ids=list(source_ids))
        finally:
            session.close()
