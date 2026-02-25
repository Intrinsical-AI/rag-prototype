# src/infrastructure/persistence/sql/models.py

import datetime
from typing import Any, ClassVar

from sqlalchemy import DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from local_rag_backend.infrastructure.persistence.sql.base import Base


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
    # Stable identity for idempotent ingestion / upsert (PR2+PR3).
    # Nullable so existing flows that only provide raw text keep working.
    external_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Optional source identifier (e.g., filename/url) for traceability and grouping.
    source_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Avoid attribute name `metadata` (reserved by SQLAlchemy declarative); keep DB column name "metadata".
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    # Hash of raw content for dedup/update decisions. Filled best-effort for legacy rows.
    content_sha256: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Optional dedup hash for chunk-level ingestion flows (PR7). Unique when present (partial index).
    chunk_dedup_sha256: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


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


class DocumentTombstone(Base):
    """
    Tombstones for deleted external_ids.

    This prevents deleted identities from reappearing after future ingestions/upserts.
    """

    __tablename__ = "document_tombstones"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    deleted_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
