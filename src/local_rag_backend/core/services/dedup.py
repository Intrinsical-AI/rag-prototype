# src/local_rag_backend/core/services/dedup.py
"""
Chunk-level dedup hash utilities.
"""

from __future__ import annotations

import hashlib


def chunk_dedup_sha256(
    *, cleaned_text: str, chunker_version: str, embedding_model_name: str
) -> str:
    """
    Compute a stable chunk-level dedup hash.

    The inputs must already be normalized (e.g., cleaned_text post cleaning).
    """
    # Use a delimiter that cannot appear in normal text to avoid accidental concatenation collisions.
    payload = "\x1f".join([cleaned_text, chunker_version, embedding_model_name])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
