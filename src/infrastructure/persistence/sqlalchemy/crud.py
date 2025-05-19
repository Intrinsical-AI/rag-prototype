# src/infrastructure/persistence/crud.py
from sqlalchemy.orm import Session

from src.infrastructure.persistence.sqlalchemy.models import Document, QaHistory


# ------------------ Docs ------------------ #
def get_documents(db: Session, ids: list[int]):
    return db.query(Document).filter(Document.id.in_(ids)).all()


def add_documents(db: Session, texts: list[str]) -> list[int | None]:
    """
    Adds multiple documents to the database from a list of text contents
    and returns a list of their assigned IDs.
    """
    if not texts:
        return []

    # 1. Crear instancias de Document
    doc_objects = [Document(content=t) for t in texts]

    # 2. Añadir todas las instancias a la sesión
    db.add_all(doc_objects)

    # 3.commit
    db.commit()

    # 4. Devolver los IDs de los documentos recién creados
    return [doc.id for doc in doc_objects]


# ------------------ History (bonus) ------------------ #
def add_history(db: Session, question: str, answer: str, source_ids=None):
    from src.infrastructure.persistence.sqlalchemy.models import QaHistory

    history = QaHistory(
        question=question,
        answer=answer,
        source_ids=source_ids if source_ids is not None else None,
    )
    db.add(history)
    db.commit()


def get_history(db: Session, limit: int = 10, offset: int = 0):
    return (
        db.query(QaHistory)
        .order_by(QaHistory.created_at.desc(), QaHistory.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def save_qa_history(db: Session, question: str, answer: str, source_ids=None):
    add_history(db, question, answer, source_ids)
