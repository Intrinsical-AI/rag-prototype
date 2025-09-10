from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: int
    content: str


Embedding = Sequence[float]
