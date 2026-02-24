# src/local_rag_backend/core/services/chunking.py
"""
Deterministic chunking utilities.

This module is intentionally small and dependency-free. Chunking is pure and
stable: given the same input text and settings, it always returns the same
chunks and boundaries.
"""

from __future__ import annotations

from local_rag_backend.core.services.schemas import TextChunk


def chunk_chars_v1(text: str, *, max_chars: int, overlap: int) -> list[TextChunk]:
    """
    Chunk `text` by fixed character windows with overlap.

    Invariants:
    - Deterministic.
    - Progress: `start_char` strictly increases when there is remaining text.
    - `overlap` is clamped to [0, max_chars-1].
    - For empty text, returns a single empty chunk at (0,0) to match current behavior.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    safe_overlap = max(0, min(int(overlap), int(max_chars) - 1))

    if len(text) <= max_chars:
        return [TextChunk(text=text, chunk_index=0, start_char=0, end_char=len(text))]

    chunks: list[TextChunk] = []
    start = 0
    idx = 0
    step = max_chars - safe_overlap
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(
            TextChunk(text=text[start:end], chunk_index=idx, start_char=start, end_char=end)
        )
        if end >= len(text):
            break
        start += step
        idx += 1

    return chunks
