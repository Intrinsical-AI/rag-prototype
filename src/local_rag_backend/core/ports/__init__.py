# src/core/ports/__init__.py
"""
Application Ports (Hex Architecture / Ports & Adapters).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from local_rag_backend.core.domain.entities import Document, Embedding, LoadedItem


# -------- Ports --------
@runtime_checkable
class EmbedderPort(Protocol):
    """Interface for embedding text into vector representations."""

    dim: int

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]: ...


@runtime_checkable
class GeneratorPort(Protocol):
    """Interface for generating text based on a question and context."""

    def generate(self, question: str, contexts: Sequence[str]) -> str: ...


@runtime_checkable
class RetrieverPort(Protocol):
    """Interface for retrieving relevant documents for a given query."""

    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]: ...


@runtime_checkable
class DocumentRepoPort(Protocol):
    """Interface for storing and retrieving documents by ID."""

    def store_documents(self, contents: Sequence[str]) -> Sequence[int]: ...
    def get(self, ids: Sequence[int]) -> Sequence[Document]: ...
    def get_all_documents(self) -> Sequence[Document]: ...


@runtime_checkable
class VectorRepoPort(Protocol):
    """Interface for storing and searching vector embeddings."""

    def upsert(self, ids: Sequence[int], vectors: Sequence[Embedding]) -> None: ...
    def similar(self, vector: Embedding, k: int) -> Sequence[tuple[int, float]]:
        """Find similar vectors, returning (ID, normalized_similarity_score)."""


@runtime_checkable
class QAHistoryPort(Protocol):
    """Interface for persisting question-answer interactions."""

    def save(self, q: str, a: str, source_ids: Sequence[int]) -> None: ...


@runtime_checkable
class LoaderPort(Protocol):
    """Interface for loading data from a source into a standard format."""

    def load(self) -> Iterable[LoadedItem]: ...
