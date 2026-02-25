# src/infrastructure/persistence/sql/crud.py
"""
CRUD operations for SQLAlchemy models.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from local_rag_backend.core.domain.types import DocId, new_doc_id
from local_rag_backend.infrastructure.persistence.sql.models import Document, QaHistory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session


def add_documents(db: Session, texts: list[str]) -> list[DocId]:
    """Store new documents and return their IDs."""
    docs = [
        Document(
            doc_id=str(new_doc_id()),
            content=text,
            content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        for text in texts
    ]
    db.add_all(docs)
    db.commit()
    return [DocId(doc.doc_id) for doc in docs]


def delete_documents(db: Session, ids: Sequence[DocId]) -> None:
    """Delete documents by IDs (best-effort rollback helper for multi-store ETL)."""
    ids_list = [str(x) for x in ids if str(x).strip()]
    if not ids_list:
        return
    db.query(Document).filter(Document.doc_id.in_(ids_list)).delete(synchronize_session=False)
    db.commit()


def add_history(
    db: Session, question: str, answer: str, source_ids: list[DocId] | None = None
) -> None:
    """Save a question-answer interaction to the history."""
    history_entry = QaHistory(
        question=question,
        answer=answer,
        source_ids=([str(x) for x in source_ids] if source_ids is not None else None),
    )
    db.add(history_entry)
    db.commit()


def get_history(db: Session, limit: int = 10, offset: int = 0) -> Sequence[QaHistory]:
    """Retrieve the most recent question-answer interactions."""
    rows: Sequence[QaHistory] = (
        db.query(QaHistory)
        .order_by(QaHistory.created_at.desc(), QaHistory.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return rows
