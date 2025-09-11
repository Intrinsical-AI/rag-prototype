from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    id: int
    content: str


@dataclass(frozen=True)
class LoadedItem:
    text: str
    metadata: Mapping[str, Any] | None = None


Embedding = Sequence[float]
