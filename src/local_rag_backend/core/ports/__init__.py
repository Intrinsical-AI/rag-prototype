from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from local_rag_backend.core.domain.entities import Document, Embedding, LoadedItem


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
    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]: ...


@runtime_checkable
class DocumentRepoPort(Protocol):
    def store_documents(self, contents: Sequence[str]) -> Sequence[int]: ...
    def get(self, ids: Sequence[int]) -> Sequence[Document]: ...
    def get_all_documents(self) -> Sequence[Document]: ...


@runtime_checkable
class VectorRepoPort(Protocol):
    def upsert(self, ids: Sequence[int], vectors: Sequence[Embedding]) -> None: ...
    # The second element must be a normalized similarity score in [0,1],
    # where 1.0 is most similar. Backends should internally convert their
    # native metric (e.g., L2 distance) into this common scale.
    def similar(self, vector: Embedding, k: int) -> Sequence[tuple[int, float]]: ...


@runtime_checkable
class QAHistoryPort(Protocol):
    def save(self, q: str, a: str, source_ids: Sequence[int]) -> None: ...


@runtime_checkable
class LoaderPort(Protocol):
    def load(self) -> Iterable[LoadedItem]: ...
