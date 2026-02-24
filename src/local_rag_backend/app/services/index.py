"""Compatibility shim for index application use-cases (moved to app.application)."""

from local_rag_backend.app.application.index import build_index_sync, rebuild_index_sync

__all__ = ["build_index_sync", "rebuild_index_sync"]
