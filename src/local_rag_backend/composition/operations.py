"""Application operations shared by command, HTTP and MCP adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from local_rag_backend.composition.search import search_live
from local_rag_backend.core.domain.retrieval import RetrievalRequest
from local_rag_backend.core.use_cases.docs_mutation import MutationCoordinator
from local_rag_backend.core.use_cases.docs_mutation_contracts import mutation_intent_from_payload

if TYPE_CHECKING:
    from local_rag_backend.composition.container import AppContainer
    from local_rag_backend.core.domain.entities import Document
    from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalResult
    from local_rag_backend.core.use_cases.results import MutationSummary
    from local_rag_backend.settings import Settings


def document_to_payload(document: Document) -> dict[str, Any]:
    return {
        "id": str(document.id),
        "content": document.content,
        "external_id": document.external_id,
        "source_id": document.source_id,
        "metadata": dict(document.metadata) if document.metadata is not None else None,
    }


def retrieval_to_payload(result: RetrievalResult) -> dict[str, Any]:
    return {
        "sources": [
            {"document": document_to_payload(document), "score": float(score)}
            for document, score in zip(result.documents, result.scores, strict=True)
        ],
        "retrieval_mode": result.mode_used,
        "backend": result.backend_used,
    }


def search(
    *, settings: Settings, question: str, k: int = 3, filters: tuple[RetrievalFilter, ...] = ()
) -> dict[str, Any]:
    request = RetrievalRequest(
        query=question,
        top_k=k,
        mode=settings.retrieval_mode,
        filters=filters,
        dual_candidate_k=settings.dual_candidate_k,
    )
    return retrieval_to_payload(search_live(settings=settings, request=request))


def mutate_documents(
    *,
    container: AppContainer,
    payload: Mapping[str, Any],
    source: str,
    require_op_id: bool = False,
) -> MutationSummary:
    """One mutation owner; the coordinator retains locking, journal and recovery."""
    intent = mutation_intent_from_payload(payload, source=source, require_op_id=require_op_id)
    container.initialize()
    bundle = container.build_docs_mutation_bundle()
    return MutationCoordinator(settings_obj=container.settings_obj, ports=bundle.ports).execute(
        intent
    )
