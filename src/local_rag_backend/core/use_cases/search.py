"""Transport-independent retrieval without generation or history effects."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_rag_backend.core.domain.retrieval import RetrievalRequest, RetrievalResult
    from local_rag_backend.core.ports import RetrieverPort


def execute_search(*, retriever: RetrieverPort, request: RetrievalRequest) -> RetrievalResult:
    """Return one retrieval result and enforce the document/score invariant."""
    result = retriever.retrieve(request)
    if len(result.documents) != len(result.scores):
        raise RuntimeError("Retriever returned different document and score counts")
    return result
