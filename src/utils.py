"""
Utils: light helpers - no external deps (nltk)
"""

import re

__all__ = ["preprocess_text"]

_HTML_TAG_RE = re.compile(r"<[^>]+>")

def preprocess_text(text: str) -> str:
    """
    Normalize texts texto:
    1. lowercase
    2. colapse whitespaces
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = _HTML_TAG_RE.sub("", text)
    return text
