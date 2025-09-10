# src/infrastructure/persistence/models.py

import datetime

from sqlalchemy import DateTime, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base


# ------------------------------------------------------------------ #
# Document ORM
# ------------------------------------------------------------------ #
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)


# ------------------------------------------------------------------ #
# BONUS: Q&A History
# ------------------------------------------------------------------ #
class QaHistory(Base):
    __tablename__ = "qa_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    source_ids: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
