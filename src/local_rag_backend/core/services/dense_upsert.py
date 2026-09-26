"""Precompute every requested vector before acquiring the mutation lock."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.ports import EmbedderPort


class UpsertItemLike(Protocol):
    @property
    def external_id(self) -> str: ...

    @property
    def content(self) -> str: ...


def precompute_vectors(
    *, items: Sequence[UpsertItemLike], embedder: EmbedderPort
) -> dict[str, list[float]]:
    """Embed all candidates: their stored content can change before the write lock."""
    if not items:
        return {}
    vectors = embedder.embed([item.content.strip() for item in items])
    if len(vectors) != len(items):
        raise RuntimeError(f"Embedder returned {len(vectors)} vectors for {len(items)} documents.")
    return {item.external_id: list(vec) for item, vec in zip(items, vectors, strict=True)}
