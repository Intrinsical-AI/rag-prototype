"""Application-facing ports for use-case decoupling from concrete adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

BlockingTaskType = Literal["default", "mutation", "network", "eval"]
ReadinessStatus = Literal["ready", "not_ready"]
T = TypeVar("T")


@dataclass(frozen=True)
class ListedDocument:
    id: str
    content: str
    external_id: str | None = None
    source_id: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class HistoryEntry:
    id: int
    question: str
    answer: str
    created_at: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReadinessDiagnostics:
    status: ReadinessStatus
    checks: dict[str, Any]


@dataclass(frozen=True)
class OpenRouterGenerateRequest:
    model: str | None
    system_instruction: str
    user_content: str
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class OpenRouterUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class OpenRouterGenerateResult:
    text: str
    usage: OpenRouterUsage | None = None


@dataclass(frozen=True)
class ImportDocsLoadResult:
    format_detected: str
    texts: tuple[str, ...]


@dataclass(frozen=True)
class EvalDatasetDocInput:
    external_id: str
    content: str
    source_id: str | None = None
    metadata: dict[str, Any] | None = None


class DocsReadPort(Protocol):
    def list_docs_page(self, *, limit: int, offset: int) -> tuple[ListedDocument, ...]: ...


class HistoryReadPort(Protocol):
    def list_history_entries(self, *, limit: int, offset: int) -> tuple[HistoryEntry, ...]: ...


class RagRuntimeFactoryPort(Protocol):
    def run_ask_eval(self, *, question: str, cfg: Any) -> dict[str, Any]: ...


class DocsImportLoaderPort(Protocol):
    def load_texts(self, *, raw: bytes) -> ImportDocsLoadResult: ...


class HealthDiagnosticsPort(Protocol):
    def ping_database(self) -> None: ...

    def get_documents_count(self) -> int: ...

    def get_history_count(self) -> int: ...

    def get_document_ids(self) -> tuple[str, ...]: ...

    def get_index_ids(self, *, id_map_path: str) -> tuple[str, ...]: ...

    def get_retrieval_index_stats(
        self,
        *,
        index_path: str,
        id_map_path: str,
        vector_backend: str,
        dim: int | None = None,
        expected_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def get_incomplete_mutation_records_count(self, *, coordination_dir: Path) -> int: ...


class OpenRouterClientPort(Protocol):
    def generate(self, *, request: OpenRouterGenerateRequest) -> OpenRouterGenerateResult: ...


class EvalStoragePort(Protocol):
    def upsert_dataset_docs(
        self,
        *,
        dataset_id: str,
        docs: tuple[EvalDatasetDocInput, ...],
    ) -> tuple[str, ...]: ...

    def list_documents(self) -> tuple[Any, ...]: ...

    def get_retriever_storage(self) -> Any: ...


class EvalRetrieverPort(Protocol):
    def retrieve(self, query: str, k: int = 5) -> tuple[tuple[Any, ...], tuple[float, ...]]: ...


class EvalRetrieverFactoryPort(Protocol):
    def build_sparse_retriever(
        self,
        *,
        storage: EvalStoragePort,
        reranker_enabled: bool,
        candidate_k: int,
        strategy: str,
    ) -> EvalRetrieverPort: ...


class BlockingExecutorPort(Protocol):
    async def run_blocking(
        self,
        func: Callable[..., T],
        /,
        *args: Any,
        task_type: BlockingTaskType = "default",
        **kwargs: Any,
    ) -> T: ...


__all__ = [
    "BlockingExecutorPort",
    "BlockingTaskType",
    "DocsImportLoaderPort",
    "DocsReadPort",
    "EvalDatasetDocInput",
    "EvalRetrieverFactoryPort",
    "EvalRetrieverPort",
    "EvalStoragePort",
    "HealthDiagnosticsPort",
    "HistoryEntry",
    "HistoryReadPort",
    "ImportDocsLoadResult",
    "ListedDocument",
    "OpenRouterClientPort",
    "OpenRouterGenerateRequest",
    "OpenRouterGenerateResult",
    "OpenRouterUsage",
    "RagRuntimeFactoryPort",
    "ReadinessDiagnostics",
    "ReadinessStatus",
]
