"""
Text normalization used across ingestion and sparse retrieval.
"""

from __future__ import annotations

import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def preprocess_text(text: str) -> str:
    """Normalize text by lowercasing, removing HTML tags, and collapsing whitespace."""
    text = text.lower().strip()
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
