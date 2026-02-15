# src/app/dependencies.py
"""FastAPI dependency helpers (re-export composition root)."""

from __future__ import annotations

from local_rag_backend.app.factory import build_rag_service, get_rag_service, reset_rag_service

__all__ = ["build_rag_service", "get_rag_service", "reset_rag_service"]
