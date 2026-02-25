from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray


class VectorEngine(Protocol):
    @property
    def backend(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def ntotal(self) -> int: ...

    def infer_dim_or_raise(self, index_path: Path) -> int: ...

    def load_or_initialize(self, index_path: Path, dim: int) -> None: ...

    def add(self, vectors: NDArray[np.float32]) -> None: ...

    def delete_keep_positions(self, keep_positions: Sequence[int]) -> None: ...

    def rebuild(self, vectors: NDArray[np.float32]) -> None: ...

    def search(
        self, query_np: NDArray[np.float32], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]: ...

    def save(self, index_path: Path) -> None: ...
