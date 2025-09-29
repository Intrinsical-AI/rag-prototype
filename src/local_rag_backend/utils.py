# src/utils.py
"""
General utility functions: text processing, data loading..
"""

from __future__ import annotations

import re

from local_rag_backend.core.ports import DocumentRepoPort

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def preprocess_text(text: str) -> str:
    """Normalize text by lowercasing, removing HTML tags, and collapsing whitespace."""
    text = text.lower().strip()
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_corpus_and_ids(doc_repo: DocumentRepoPort) -> tuple[list[str], list[int]]:
    """Fetch all documents from a repository and separate contents from IDs."""
    docs = doc_repo.get_all_documents()
    return [d.content for d in docs], [d.id for d in docs]
