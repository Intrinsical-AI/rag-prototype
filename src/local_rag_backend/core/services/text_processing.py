"""
Text normalization used across ingestion and sparse retrieval.
"""

from __future__ import annotations

import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def preprocess_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_html: bool = True,
    collapse_whitespace: bool = True,
    strip: bool = True,
) -> str:
    """Normalize text for ingestion/retrieval (configurable, deterministic)."""
    out = text
    if strip:
        out = out.strip()
    if lowercase:
        out = out.lower()
    if remove_html:
        out = _HTML_TAG_RE.sub(" ", out)
    if collapse_whitespace:
        out = re.sub(r"\s+", " ", out).strip()
    return out
