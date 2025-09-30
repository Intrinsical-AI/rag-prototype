"""
Intrinsical-AI RAG Prototype
Copyright (c) 2025 Intrinsical-AI

Module: Core Domain Entities
Purpose: Defines the fundamental domain entities and value objects for the RAG system.
         These entities represent the core business concepts independent of infrastructure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    """Core domain entity representing a text document in the RAG system.

    This immutable entity encapsulates the essential properties of a document
    that can be stored, indexed, and retrieved during RAG operations.

    Attributes:
        id: Unique identifier for the document
        content: The actual text content of the document
    """

    id: int
    content: str


@dataclass(frozen=True)
class LoadedItem:
    """Represents a raw item loaded from external sources before processing.

    This entity serves as an intermediate representation between raw data sources
    (CSV, web pages, files) and processed documents ready for indexing.

    Attributes:
        text: The raw text content extracted from the source
        metadata: Optional key-value pairs containing additional information
                 about the source (e.g., URL, file path, timestamps)
    """

    text: str
    metadata: Mapping[str, Any] | None = None


# --- Type Aliases ---

Embedding = Sequence[float]
"""Type alias for vector embeddings used in dense retrieval operations."""
