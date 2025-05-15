"""
File: src/db/crud.py
Database CRUD operations for documents and history.
"""

from sqlalchemy.orm import Session
from src.db import models


# ------------------ Docs ------------------ #
def get_documents(db: Session, ids: list[int]):
    return db.query(models.Document).filter(models.Document.id.in_(ids)).all()


def add_documents(db: Session, texts: list[str]):
    for t in texts:
        db.add(models.Document(content=t))
    db.commit()


# ------------------ History (bonus) ------------------ #
def add_history(db: Session, question: str, answer: str):
    db.add(models.QaHistory(question=question, answer=answer))
    db.commit()


def get_history(db: Session, limit: int = 10, offset: int = 0):
    return (
        db.query(models.QaHistory)
        .order_by(
            models.QaHistory.created_at.desc(), models.QaHistory.id.desc()
        )  # <--- 2nd ordering by ID
        .offset(offset)
        .limit(limit)
        .all()
    )


def save_qa_history(db: Session, question: str, answer: str):
    models.Base.metadata.create_all(bind=db.get_bind())
    add_history(db, question, answer)
