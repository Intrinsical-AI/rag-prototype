from __future__ import annotations

import importlib.util
from typing import cast

from local_rag_backend.infrastructure.persistence.vector.engines.base import VectorEngine
from local_rag_backend.infrastructure.persistence.vector.engines.faiss_engine import FaissEngine
from local_rag_backend.infrastructure.persistence.vector.engines.numpy_engine import NumpyEngine


def is_faiss_available() -> bool:
    return importlib.util.find_spec("faiss") is not None


def build_vector_engine(*, backend: str) -> VectorEngine:
    resolved = str(backend).strip().lower()

    if resolved == "auto":
        if is_faiss_available():
            return cast(VectorEngine, FaissEngine())
        return cast(VectorEngine, NumpyEngine())
    if resolved == "faiss":
        return cast(VectorEngine, FaissEngine())
    if resolved == "numpy":
        return cast(VectorEngine, NumpyEngine())

    raise ValueError(f"Unsupported vector backend: {backend!r}. Expected auto|faiss|numpy")
