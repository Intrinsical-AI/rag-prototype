# src/core/domain/entities.py
"""
Domain entities for the application.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import datetime


@dataclass(frozen=True)
class Document:
    id: int
    content: str
    # Optional stable identity / traceability fields (may be None for legacy docs).
    external_id: str | None = None
    source_id: str | None = None
    metadata: Mapping[str, Any] | None = None
    content_sha256: str | None = None
    created_at: datetime.datetime | None = None
    updated_at: datetime.datetime | None = None


@dataclass(frozen=True)
class LoadedItem:
    text: str
    metadata: Mapping[str, Any] | None = None


Embedding = Sequence[float]
