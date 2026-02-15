# src/infrastructure/persistence/sqlalchemy/models.py

import datetime
from typing import ClassVar

from sqlalchemy import DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base


class Document(Base):
    """ORM model for a text document."""

    __tablename__ = "documents"
    # Critical: without SQLite AUTOINCREMENT, INTEGER PRIMARY KEY values can be reused after
    # deletes (e.g., insert id=1, delete all rows, next insert may return id=1 again).
    # In a multi-store setup (SQL + FAISS), ID reuse can make an old vector point to new
    # content (or vice versa), causing data integrity issues and potential content leakage.
    __table_args__: ClassVar[dict[str, bool]] = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)


class QaHistory(Base):
    """ORM model for a question-answer interaction history."""

    __tablename__ = "qa_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    source_ids: Mapped[list[int] | None] = mapped_column(JSON)
