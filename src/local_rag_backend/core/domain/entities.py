# src/local_rag_backend/core/domain/entities.py
"""
Domain entities for the application.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from local_rag_backend.core.domain.types import DocId, ItemLineage


@dataclass(frozen=True)
class Document:
    id: DocId
    content: str
    # Optional stable source identity for idempotent upsert semantics.
    external_id: str | None = None
    source_id: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LoadedItem:
    text: str
    lineage: ItemLineage
    metadata: Mapping[str, Any] | None = None


Embedding = Sequence[float]
