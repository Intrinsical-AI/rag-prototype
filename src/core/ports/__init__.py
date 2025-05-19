from __future__ import annotations

from typing import Protocol, Sequence, Tuple, runtime_checkable

from src.core.domain.entities import Document, Embedding


# -------- Ports --------
@runtime_checkable
class EmbedderPort(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]: ...


@runtime_checkable
class GeneratorPort(Protocol):
    def generate(self, question: str, contexts: Sequence[str]) -> str: ...


@runtime_checkable
class RetrieverPort(Protocol):
    def retrieve(
        self, query: str, k: int = 5
    ) -> Tuple[Sequence[Document], Sequence[float]]: ...


@runtime_checkable
class DocumentRepoPort(Protocol):
    def save(self, contents: Sequence[str]) -> Sequence[int]: ...
    def get(self, ids: Sequence[int]) -> Sequence[Document]: ...


@runtime_checkable
class VectorRepoPort(Protocol):
    def upsert(self, ids: Sequence[int], vectors: Sequence[Embedding]) -> None: ...
    def similar(self, vector: Embedding, k: int) -> Sequence[tuple[int, float]]: ...


@runtime_checkable
class QAHistoryPort(Protocol):
    def save(self, q: str, a: str, source_ids: Sequence[int]) -> None: ...
