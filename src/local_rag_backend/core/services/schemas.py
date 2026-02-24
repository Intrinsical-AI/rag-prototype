from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


# --- CHUNKING ---
@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int  # exclusive


# --- EVALUATION ---
@dataclass(frozen=True)
class EvalDoc:
    external_id: str
    content: str
    source_id: str | None = None


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_external_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvalDataset:
    dataset_id: str
    schema_version: int
    docs: tuple[EvalDoc, ...]
    queries: tuple[EvalQuery, ...]


@dataclass(frozen=True)
class EvalResult:
    dataset_id: str
    retrieval_mode: str
    reranker_enabled: bool
    k: int
    queries: int
    hit_rate: float
    mrr: float


# ---Re-Ranking ---


def _tokens(text: str) -> set[str]:
    from local_rag_backend.core.services.text_processing import preprocess_text

    # Keep tokenization aligned with the sparse retriever (preprocess_text + \\w+).
    return set(re.findall(r"\w+", preprocess_text(text)))


@dataclass(frozen=True)
class OverlapV1Reranker:
    """
    Token overlap reranker (cheap heuristic).

    Score: |tokens(query) ∩ tokens(doc)| / max(1, |tokens(query)|)
    """

    def score(self, *, query: str, doc_text: str) -> float:
        qt = _tokens(query)
        if not qt:
            return 0.0
        dt = _tokens(doc_text)
        if not dt:
            return 0.0
        return float(len(qt & dt) / max(1, len(qt)))


# -- INFRA, Loader Factory --
DetectedFormat = Literal[
    "csv", "markdown", "text", "binary", "unknown", "chatgpt_export", "gemini_export"
]


@dataclass(frozen=True)
class Detection:
    fmt: DetectedFormat
    reason: str
    mime: str | None = None
