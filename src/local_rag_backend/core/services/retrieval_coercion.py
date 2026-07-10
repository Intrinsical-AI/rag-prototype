"""Shared coercion for retriever outputs accepted by eval/use-case boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from local_rag_backend.core.domain.entities import Document as DomainDocument
from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    retrieval_result_from_pairs,
)


def coerce_retrieval_result(
    raw_result: Any,
    *,
    request: RetrievalRequest,
    backend_used: str = "eval",
) -> RetrievalResult:
    if isinstance(raw_result, RetrievalResult):
        return raw_result
    if (
        isinstance(raw_result, tuple)
        and len(raw_result) == 2
        and isinstance(raw_result[0], (list, tuple))
        and isinstance(raw_result[1], (list, tuple))
    ):
        docs, scores = raw_result
        return retrieval_result_from_pairs(
            docs=cast("Sequence[DomainDocument]", docs),
            scores=cast("Sequence[float]", scores),
            mode_used=request.mode,
            backend_used=backend_used,
        )
    raise RuntimeError(f"Unsupported eval retriever response type: {type(raw_result)!r}")
