from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Document:
    id: int
    content: str


Embedding = Sequence[float]
