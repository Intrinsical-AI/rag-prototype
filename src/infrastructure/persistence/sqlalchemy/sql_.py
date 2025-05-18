# src/adapters/storage/sql_crud.py

from typing import Sequence

from sqlalchemy.orm import sessionmaker

from src.core.domain.entities import Document as DomainDocument
from src.core.ports import DocumentRepoPort, QAHistoryPort
from src.infrastructure.persistence.sqlalchemy.base import SessionLocal
from src.infrastructure.persistence.sqlalchemy.crud import (
    add_documents,
    save_qa_history,
)
from src.infrastructure.persistence.sqlalchemy.models import Document as DbDocument

# src/infrastructure/persistence/sqlalchemy/sql_.py


class SqlDocumentStorage(DocumentRepoPort):
    def __init__(self, session_factory: sessionmaker = SessionLocal):
        self._session_factory = session_factory

    def store_documents(self, texts: Sequence[str]) -> Sequence[int]:
        session = self._session_factory()
        try:
            return add_documents(session, list(texts))
        finally:
            session.close()

    def get(self, ids: Sequence[int]) -> Sequence[DomainDocument]:
        session = self._session_factory()
        try:
            db_docs = session.query(DbDocument).filter(DbDocument.id.in_(ids)).all()
            return [DomainDocument(id=d.id, content=d.content) for d in db_docs]
        finally:
            session.close()

    def get_all_documents(self) -> Sequence[DomainDocument]:
        session = self._session_factory()
        try:
            db_docs = session.query(DbDocument).order_by(DbDocument.id).all()
            return [DomainDocument(id=d.id, content=d.content) for d in db_docs]
        finally:
            session.close()

    save = store_documents


class HistorySqlStorage(QAHistoryPort):
    def save(self, q, a, source_ids):
        session = SessionLocal()
        try:
            save_qa_history(session, q, a, source_ids=source_ids)
        finally:
            session.close()
