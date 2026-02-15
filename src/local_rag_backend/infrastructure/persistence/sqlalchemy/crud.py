# src/infrastructure/persistence/sqlalchemy/crud.py
"""
CRUD operations for SQLAlchemy models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.infrastructure.persistence.sqlalchemy.models import Document, QaHistory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session


def get_documents(db: Session, ids: list[int]) -> Sequence[Document]:
    """Retrieve documents by their IDs."""
    return db.query(Document).filter(Document.id.in_(ids)).all()


def add_documents(db: Session, texts: list[str]) -> list[int]:
    """Store new documents and return their IDs."""
    docs = [Document(content=text) for text in texts]
    db.add_all(docs)
    db.commit()
    return [doc.id for doc in docs]


def delete_documents(db: Session, ids: list[int]) -> None:
    """Delete documents by IDs (best-effort rollback helper for multi-store ETL)."""
    if not ids:
        return
    db.query(Document).filter(Document.id.in_(ids)).delete(synchronize_session=False)
    db.commit()


def add_history(
    db: Session, question: str, answer: str, source_ids: list[int] | None = None
) -> None:
    """Save a question-answer interaction to the history."""
    history_entry = QaHistory(question=question, answer=answer, source_ids=source_ids)
    db.add(history_entry)
    db.commit()


def get_history(db: Session, limit: int = 10, offset: int = 0) -> Sequence[QaHistory]:
    """Retrieve the most recent question-answer interactions."""
    return (
        db.query(QaHistory)
        .order_by(QaHistory.created_at.desc(), QaHistory.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
