"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

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


def preprocess_text(text: str | None) -> str:
    """Normalize text for consistent processing with robust input validation.

    Applies standard text normalization:
    - Validates input type and handles None gracefully
    - Converts to lowercase
    - Removes HTML tags (replaced with spaces to prevent word concatenation)
    - Collapses multiple whitespace characters
    - Strips leading/trailing whitespace

    Args:
        text: Raw text to normalize (can be None)

    Returns:
        Normalized text ready for tokenization or embedding.
        Returns empty string if input is None or invalid.

    Note:
        This function is designed to be defensive and never crash,
        making it safe to use in pipelines where text might be None.
    """
    # --- Input Validation ---
    if text is None:
        return ""
    
    if not isinstance(text, str):
        # Convert to string if possible, otherwise return empty
        try:
            text = str(text)
        except Exception:
            return ""
    
    # --- Text Normalization ---
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
