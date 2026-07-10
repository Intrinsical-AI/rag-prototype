"""Retrieval request/result contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from local_rag_backend.core.domain.entities import Document

RetrievalMode = Literal["sparse", "dense", "dual", "hybrid"]
TOP_LEVEL_FILTER_FIELDS = frozenset({"scope", "snapshot_id", "source_id"})
METADATA_FILTER_PREFIX = "metadata."


def normalize_filter_field(field: str) -> str:
    normalized = str(field).strip()
    if not normalized:
        raise ValueError("RetrievalFilter.field must not be blank")
    if normalized in TOP_LEVEL_FILTER_FIELDS:
        return normalized
    if normalized.startswith(METADATA_FILTER_PREFIX):
        metadata_key = normalized[len(METADATA_FILTER_PREFIX) :].strip()
        if metadata_key and all(part.strip() for part in metadata_key.split(".")):
            return normalized
    raise ValueError(
        "RetrievalFilter.field must be one of scope, snapshot_id, source_id, or metadata.<key>"
    )


def metadata_key_for_filter_field(field: str) -> str | None:
    normalized = normalize_filter_field(field)
    if normalized in TOP_LEVEL_FILTER_FIELDS:
        return None
    return normalized[len(METADATA_FILTER_PREFIX) :]


def normalize_filter_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value if item is not None and str(item).strip())
    rendered = str(value).strip()
    return (rendered,) if rendered else ()


def document_field_values(document: Document, field: str) -> tuple[str, ...]:
    if field == "source_id":
        return normalize_filter_values(document.source_id)
    metadata = dict(document.metadata or {})
    if field in {"scope", "snapshot_id"}:
        return normalize_filter_values(metadata.get(field))
    metadata_key = metadata_key_for_filter_field(field)
    if metadata_key is None:
        return ()
    value: Any = metadata
    for part in metadata_key.split("."):
        if not isinstance(value, dict):
            return ()
        value = value.get(part)
    return normalize_filter_values(value)


def document_matches_filters(document: Document, filters: Sequence[RetrievalFilter]) -> bool:
    if not filters:
        return True
    for filter_item in filters:
        values = document_field_values(document, filter_item.field)
        if not values or not any(value in filter_item.values for value in values):
            return False
    return True


@dataclass(frozen=True)
class RetrievalFilter:
    """Structured filter applied during retrieval."""

    field: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = tuple(str(value).strip() for value in self.values if str(value).strip())
        field = normalize_filter_field(self.field)
        if not normalized:
            raise ValueError("RetrievalFilter.values must not be empty")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "values", normalized)


@dataclass(frozen=True)
class RetrievalRequest:
    """Single retrieval plan for one question."""

    query: str
    top_k: int = 5
    mode: RetrievalMode = "sparse"
    filters: tuple[RetrievalFilter, ...] = ()
    candidate_k: int | None = None
    dual_candidate_k: int | None = None
    min_score: float | None = None

    def __post_init__(self) -> None:
        query = str(self.query).strip()
        if not query:
            raise ValueError("RetrievalRequest.query must not be blank")
        if str(self.mode) not in {"sparse", "dense", "dual", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {self.mode}")
        if int(self.top_k) <= 0:
            raise ValueError("RetrievalRequest.top_k must be positive")
        if self.candidate_k is not None and int(self.candidate_k) <= 0:
            raise ValueError("RetrievalRequest.candidate_k must be positive when set")
        if self.dual_candidate_k is not None and int(self.dual_candidate_k) <= 0:
            raise ValueError("RetrievalRequest.dual_candidate_k must be positive when set")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "top_k", int(self.top_k))
        if self.candidate_k is not None:
            object.__setattr__(self, "candidate_k", int(self.candidate_k))
        if self.dual_candidate_k is not None:
            object.__setattr__(self, "dual_candidate_k", int(self.dual_candidate_k))


@dataclass(frozen=True)
class RetrievedDoc:
    """Document plus retrieval metadata."""

    document: Document
    score: float
    stage: str | None = None
    score_breakdown: Mapping[str, float] | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """Structured retrieval output."""

    items: tuple[RetrievedDoc, ...]
    mode_used: RetrievalMode
    backend_used: str
    candidate_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def documents(self) -> tuple[Document, ...]:
        return tuple(item.document for item in self.items)

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(float(item.score) for item in self.items)


def retrieval_result_from_pairs(
    *,
    docs: Sequence[Document],
    scores: Sequence[float],
    mode_used: RetrievalMode,
    backend_used: str,
    stage: str | None = None,
    candidate_count: int | None = None,
) -> RetrievalResult:
    items = tuple(
        RetrievedDoc(document=doc, score=float(score), stage=stage)
        for doc, score in zip(docs, scores, strict=False)
    )
    return RetrievalResult(
        items=items,
        mode_used=mode_used,
        backend_used=backend_used,
        candidate_count=len(items) if candidate_count is None else int(candidate_count),
    )
