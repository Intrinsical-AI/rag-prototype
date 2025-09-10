"""
Utils: light helpers - no external deps (nltk)
"""

import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def preprocess_text(text: str) -> str:
    """
    Normalize text:
    1. lowercase
    2. collapse whitespaces
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = _HTML_TAG_RE.sub("", text)
    return text


def get_corpus_and_ids(doc_repo: DocumentRepoPort) -> tuple[list[str], list[int]]:
    docs = doc_repo.get_all_documents()
    return [d.content for d in docs], [d.id for d in docs]
