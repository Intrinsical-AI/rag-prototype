"""
File: src/db/models.py
Path: src/db/models.py
SQLAlchemy ORM models for the database.
"""

from sqlalchemy import Column, Integer, Text, DateTime, func
from src.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)


# ------------------------------------------------------------------ #
# BONUS: Q&A History
# ------------------------------------------------------------------ #
class QaHistory(Base):
    __tablename__ = "qa_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
