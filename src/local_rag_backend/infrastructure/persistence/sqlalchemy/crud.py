# src/infrastructure/persistence/crud.py
from collections.abc import Sequence

from sqlalchemy.orm import Session

from local_rag_backend.infrastructure.persistence.sqlalchemy.models import Document, QaHistory


# ------------------ Docs ------------------ #
def get_documents(db: Session, ids: list[int]) -> Sequence[Document]:
    return db.query(Document).filter(Document.id.in_(ids)).all()  # type: ignore[no-any-return]


def add_documents(db: Session, texts: list[str]) -> list[int]:
    docs = [Document(content=text) for text in texts]
    db.add_all(docs)
    db.commit()
    return [doc.id for doc in docs]


# ------------------ History (bonus) ------------------ #
def add_history(
    db: Session, question: str, answer: str, source_ids: list[int] | None = None
) -> None:
    from local_rag_backend.infrastructure.persistence.sqlalchemy.models import QaHistory

    history = QaHistory(
        question=question,
        answer=answer,
        source_ids=source_ids if source_ids is not None else None,
    )
    db.add(history)
    db.commit()


def get_history(db: Session, limit: int = 10, offset: int = 0) -> Sequence[QaHistory]:
    return (
        db.query(QaHistory)
        .order_by(QaHistory.created_at.desc(), QaHistory.id.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )


def save_qa_history(
    db: Session, question: str, answer: str, source_ids: list[int] | None = None
) -> None:
    add_history(db, question, answer, source_ids)
