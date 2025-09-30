"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Module: SQLAlchemy ORM Models
Purpose: Defines database models for document storage and Q&A history persistence.
         Implements the data layer using SQLAlchemy ORM with proper typing support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base

if TYPE_CHECKING:
    import datetime


class Document(Base):
    """ORM model representing a text document in the database.

    This model stores the core document content that can be indexed and retrieved
    during RAG operations. Each document has a unique identifier and text content.

    Attributes:
        id: Primary key, auto-incrementing unique identifier
        content: Full text content of the document (stored as TEXT for large content)
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)


class QaHistory(Base):
    """ORM model for storing question-answer interaction history.

    This model provides audit trail and analytics capabilities by persisting
    all user interactions with the RAG system, including source document references.

    Attributes:
        id: Primary key, auto-incrementing unique identifier
        question: The user's original question text
        answer: The generated response from the RAG system
        created_at: Timestamp of the interaction (timezone-aware, server default)
        source_ids: JSON array of document IDs used to generate the answer
    """

    __tablename__ = "qa_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    source_ids: Mapped[list[int] | None] = mapped_column(JSON)
