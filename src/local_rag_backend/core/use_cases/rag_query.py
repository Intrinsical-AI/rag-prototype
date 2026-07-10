"""Application orchestration for RAG eval/history flows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from local_rag_backend.core.ports import HistoryEntry, HistoryReadPort, RagRuntimeFactoryPort


class AskEvalConfigLike(Protocol):
    """Structural contract for per-request RAG evaluation config.

    Transport-neutral: satisfied by any object that exposes these fields,
    including Pydantic HTTP schemas and plain dataclasses.
    """

    retrieval_mode: str
    filters: object | None
    k: int
    dual_candidate_k: int | None
    hybrid_alpha: float | None
    llm_provider: str | None
    model: str | None
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    prompt_template: str | None


@dataclass(frozen=True)
class AskEvalOutcome:
    rag_result: dict[str, Any]
    latency_ms: int


def execute_ask_eval_sync(
    *,
    question: str,
    cfg: AskEvalConfigLike,
    rag_runtime_factory: RagRuntimeFactoryPort,
) -> AskEvalOutcome:
    """Run one ephemeral ask-eval operation and return result + latency."""
    t0 = time.perf_counter()
    rag_result = rag_runtime_factory.run_ask_eval(question=question, cfg=cfg)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return AskEvalOutcome(rag_result=rag_result, latency_ms=latency_ms)


def list_history_entries_sync(
    *,
    history_reader: HistoryReadPort,
    limit: int,
    offset: int,
) -> tuple[HistoryEntry, ...]:
    """Fetch persisted Q/A history rows from storage."""
    return history_reader.list_history_entries(limit=limit, offset=offset)
