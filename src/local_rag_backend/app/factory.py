"""
Composition root (hex architecture).

Historically this project wired dependencies in `app/dependencies.py`.  The docs reference
`app/factory.py` as the composition root, so this module exists as a stable import path.
"""

from __future__ import annotations

from local_rag_backend.app.dependencies import get_rag_service, reset_rag_service

__all__ = ["get_rag_service", "reset_rag_service"]
