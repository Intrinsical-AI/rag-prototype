"""
Utils: light helpers - no external deps (nltk)
"""

import re
from collections.abc import Sequence

from local_rag_backend.core.ports import DocumentRepoPort

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def preprocess_text(text: str) -> str:
    """
    Normalize text:
    1. lowercase
    2. remove HTML tags (replace with space to prevent word concatenation)
    3. collapse whitespaces
    """
    text = text.lower().strip()
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_similarities_from_distances(dists: Sequence[float]) -> list[float]:
    """
    Convert distances to normalized similarities using linear inverse mapping.

    Maps distance values to similarity scores where smaller distances result in higher similarities.
    Uses min-max normalization to ensure output is in [0, 1] range.

    Args:
        dists: Sequence of distance values (lower is better)

    Returns:
        List of normalized similarity scores in [0, 1] range (higher is better)
    """
    if not dists:
        return []

    if len(dists) == 1:
        return [1.0]

    # Find min and max distances
    min_dist, max_dist = min(dists), max(dists)

    # If all distances are the same, return all 1.0
    if max_dist == min_dist:
        return [1.0] * len(dists)

    # Normalize: smaller distance = higher similarity
    # Map [min_dist, max_dist] to [1.0, 0.0]
    return [(max_dist - d) / (max_dist - min_dist) for d in dists]


def get_corpus_and_ids(doc_repo: DocumentRepoPort) -> tuple[list[str], list[int]]:
    docs = doc_repo.get_all_documents()
    return [d.content for d in docs], [d.id for d in docs]
