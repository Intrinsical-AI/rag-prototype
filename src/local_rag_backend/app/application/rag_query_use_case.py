"""Application orchestration for RAG eval/history flows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from local_rag_backend.core.services.rag_runtime import RagService
from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import (
    HistorySqlStorage,
    SqlDocumentStorage,
)
from local_rag_backend.infrastructure.persistence.sql.crud import get_history

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.orm import Session

    from local_rag_backend.core.ports import (
        DocumentRepoPort,
        GeneratorPort,
        QAHistoryPort,
        RetrieverPort,
    )


class AskEvalConfigLike(Protocol):
    @property
    def k(self) -> int: ...


@dataclass(frozen=True)
class AskEvalOutcome:
    rag_result: dict[str, Any]
    latency_ms: int


def execute_ask_eval_sync(
    *,
    question: str,
    cfg: AskEvalConfigLike,
    build_retriever_from_config: Callable[..., RetrieverPort],
    build_generator_from_config: Callable[[AskEvalConfigLike], GeneratorPort],
    doc_repo_factory: Callable[[], DocumentRepoPort] = cast(
        "Callable[[], DocumentRepoPort]", SqlDocumentStorage
    ),
    history_repo_factory: Callable[[], QAHistoryPort] = cast(
        "Callable[[], QAHistoryPort]", HistorySqlStorage
    ),
    rag_service_factory: Callable[..., RagService] = RagService,
) -> AskEvalOutcome:
    """Run one ephemeral ask-eval operation and return result + latency."""
    doc_repo = doc_repo_factory()
    docs = doc_repo.get_all_documents()
    retriever = build_retriever_from_config(cfg, doc_repo, preloaded_docs=docs)
    generator = build_generator_from_config(cfg)

    history_storage = history_repo_factory()
    service = rag_service_factory(retriever, generator, history_storage)
    t0 = time.perf_counter()
    rag_result = service.ask(question=question, top_k=cfg.k)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return AskEvalOutcome(rag_result=rag_result, latency_ms=latency_ms)


def list_history_entries_sync(*, db: Session, limit: int, offset: int) -> list[Any]:
    """Fetch persisted Q/A history rows from storage."""
    return list(get_history(db=db, limit=limit, offset=offset))
