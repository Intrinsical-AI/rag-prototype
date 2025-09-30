"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Module: Core Ports
Purpose: Defines port interfaces for hexagonal architecture implementation.
         Provides abstractions for external dependencies and infrastructure adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from local_rag_backend.core.domain.entities import Document, Embedding, LoadedItem


# --- Port Interfaces ---


@runtime_checkable
class EmbedderPort(Protocol):
    """Port for converting text to vector embeddings."""

    dim: int
    """Vector dimension produced by this embedder."""

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]:
        """Convert text sequences to vector embeddings."""


@runtime_checkable
class GeneratorPort(Protocol):
    """Port for LLM-based text generation."""

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """Generate answer from question and context documents."""


@runtime_checkable
class RetrieverPort(Protocol):
    """Port for document retrieval based on query similarity."""

    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]:
        """Retrieve top-k documents with similarity scores."""


@runtime_checkable
class DocumentRepoPort(Protocol):
    """Port for document storage and retrieval operations."""

    def store_documents(self, contents: Sequence[str]) -> Sequence[int]:
        """Store documents and return assigned IDs."""

    def get(self, ids: Sequence[int]) -> Sequence[Document]:
        """Retrieve documents by their IDs."""

    def get_all_documents(self) -> Sequence[Document]:
        """Retrieve all stored documents."""


@runtime_checkable
class VectorRepoPort(Protocol):
    """Port for vector storage and similarity search operations."""

    def upsert(self, ids: Sequence[int], vectors: Sequence[Embedding]) -> None:
        """Insert or update vectors in the index."""

    def similar(self, vector: Embedding, k: int) -> Sequence[tuple[int, float]]:
        """Find similar vectors, returning (ID, similarity_score) pairs."""


@runtime_checkable
class QAHistoryPort(Protocol):
    """Port for persisting question-answer interaction history."""

    def save(self, q: str, a: str, source_ids: Sequence[int]) -> None:
        """Save Q&A interaction with source document references."""


@runtime_checkable
class LoaderPort(Protocol):
    """Port for loading data from external sources."""

    def load(self) -> Iterable[LoadedItem]:
        """Load and yield data items from the source."""
