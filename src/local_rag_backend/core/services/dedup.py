# src/local_rag_backend/core/services/dedup.py
"""
Chunk-level dedup hash utilities.
"""

from __future__ import annotations

import hashlib


def chunk_dedup_sha256(*, text: str, chunker_version: str) -> str:
    """
    Compute a stable chunk-level dedup hash.

    Chunk identity depends on stored text and chunker version, never on embedding mode.
    """
    # Use a delimiter that cannot appear in normal text to avoid accidental concatenation collisions.
    payload = "\x1f".join([text, chunker_version])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
