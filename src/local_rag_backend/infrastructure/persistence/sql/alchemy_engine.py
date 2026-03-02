"""Backward-compatible import surface for SQLAlchemy repository adapters."""

from __future__ import annotations

from local_rag_backend.infrastructure.persistence.sql.document_storage import SqlDocumentStorage
from local_rag_backend.infrastructure.persistence.sql.history_storage import HistorySqlStorage
from local_rag_backend.infrastructure.persistence.sql.sessions import (
    get_managed_session,
    get_session,
)
from local_rag_backend.infrastructure.persistence.sql.system_state import SystemStateStorage

__all__ = [
    "HistorySqlStorage",
    "SqlDocumentStorage",
    "SystemStateStorage",
    "get_managed_session",
    "get_session",
]
