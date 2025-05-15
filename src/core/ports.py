"""
File: src/core/ports.py
Domain Interfaces - Adapters
"""

from typing import Protocol, List, Tuple, runtime_checkable


@runtime_checkable
class RetrieverPort(Protocol):
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        ...


@runtime_checkable
class GeneratorPort(Protocol):
    def generate(self, question: str, contexts: List[str]) -> str:
        ...


@runtime_checkable
class EmbedderPort(Protocol):
    DIM: int

    def embed(self, text: str) -> List[float]:
        ...
