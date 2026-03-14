"""Core service-layer data types (transport-agnostic DTOs)."""

from __future__ import annotations

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
    ndcg_at_k: float
    map_at_k: float
    mrr_at_k: float
    precision_at_k: float
    recall_at_k: float


# --- INFRASTRUCTURE INGESTION DETECTION ---
DetectedFormat = Literal[
    "csv", "markdown", "text", "binary", "unknown", "chatgpt_export", "gemini_export"
]


@dataclass(frozen=True)
class Detection:
    fmt: DetectedFormat
    reason: str
    mime: str | None = None
