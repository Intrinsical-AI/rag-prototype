from collections.abc import Sequence
from dataclasses import dataclass
from typing import Mapping, Any


@dataclass(frozen=True)
class Document:
    id: int
    content: str


@dataclass(frozen=True)
class LoadedItem:
    text: str
    metadata: Mapping[str, Any] | None = None


Embedding = Sequence[float]
