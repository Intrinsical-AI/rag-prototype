"""
Backward-compatible re-export of miscellaneous helpers.

Deprecated: import from `local_rag_backend.core.services.text_processing` and
`local_rag_backend.core.services.corpus` instead.
"""

from __future__ import annotations

from local_rag_backend.core.services.corpus import get_corpus_and_ids
from local_rag_backend.core.services.text_processing import preprocess_text

__all__ = ["get_corpus_and_ids", "preprocess_text"]
