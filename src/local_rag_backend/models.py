"""
Backward-compatible re-export of HTTP API schemas.

Deprecated: import from `local_rag_backend.app.schemas` instead.

Rationale:
- Keeping transport schemas in the app layer avoids long-term coupling and circular imports.
- This shim preserves existing imports during the refactor.
"""

from __future__ import annotations

from local_rag_backend.app.schemas import (
    AskEvalConfig,
    AskEvalRequest,
    AskEvalResponse,
    AskRequest,
    AskResponse,
    DeleteDocsRequest,
    DeleteDocsResponse,
    DocumentInDB,
    HistoryItem,
    QueryResult,
    RebuildIndexResponse,
    UpsertDocItem,
    UpsertDocResult,
    UpsertDocsRequest,
    UpsertDocsResponse,
)

__all__ = [
    "AskEvalConfig",
    "AskEvalRequest",
    "AskEvalResponse",
    "AskRequest",
    "AskResponse",
    "DeleteDocsRequest",
    "DeleteDocsResponse",
    "DocumentInDB",
    "HistoryItem",
    "QueryResult",
    "RebuildIndexResponse",
    "UpsertDocItem",
    "UpsertDocResult",
    "UpsertDocsRequest",
    "UpsertDocsResponse",
]
