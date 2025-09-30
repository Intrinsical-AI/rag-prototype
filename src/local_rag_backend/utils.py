"""
Intrinsical-AI RAG Prototype
Copyright (c) 2025 Intrinsical-AI

Module: Utility Functions
Purpose: Common utility functions for text processing and data manipulation.
         Provides reusable helpers across the application.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_rag_backend.core.ports import DocumentRepoPort

# --- Text Processing ---

_HTML_TAG_RE = re.compile(r"<[^>]+>")
"""Compiled regex for HTML tag removal."""


def preprocess_text(text: str) -> str:
    """Normalize text for consistent processing.

    Applies standard text normalization:
    - Converts to lowercase
    - Removes HTML tags (replaced with spaces to prevent word concatenation)
    - Collapses multiple whitespace characters
    - Strips leading/trailing whitespace

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text ready for tokenization or embedding
    """
    text = text.lower().strip()
    text = _HTML_TAG_RE.sub(" ", text)  # Replace tags with spaces
    text = re.sub(r"\s+", " ", text).strip()  # Collapse whitespace
    return text


# --- Data Helpers ---


def get_corpus_and_ids(doc_repo: DocumentRepoPort) -> tuple[list[str], list[int]]:
    """Extract document contents and IDs from repository.

    Convenience function for retrievers that need both document content
    and IDs in separate lists for processing.

    Args:
        doc_repo: Document repository to query

    Returns:
        Tuple of (content_list, id_list) in corresponding order
    """
    docs = doc_repo.get_all_documents()
    return [d.content for d in docs], [d.id for d in docs]
