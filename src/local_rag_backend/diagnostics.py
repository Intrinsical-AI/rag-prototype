"""
Backward-compatible re-export of app diagnostics helpers.

Deprecated: import from `local_rag_backend.app.diagnostics` instead.
"""

from __future__ import annotations

from local_rag_backend.app.diagnostics import (
    get_document_ids,
    get_documents_count,
    get_history_count,
    get_retrieval_index_stats,
)

__all__ = [
    "get_document_ids",
    "get_documents_count",
    "get_history_count",
    "get_retrieval_index_stats",
]
