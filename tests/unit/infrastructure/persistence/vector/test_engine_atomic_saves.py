from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from local_rag_backend.infrastructure.persistence.shared.atomic_io import atomic_replace_file
from local_rag_backend.infrastructure.persistence.vector.engines import faiss_engine, numpy_engine

if TYPE_CHECKING:
    from types import ModuleType


class _FakeFaiss:
    def write_index(self, _index: object, path: str) -> None:
        Path(path).write_bytes(b"new-faiss-index")


def _engine_case(
    backend: str, index_path: Path
) -> tuple[faiss_engine.FaissEngine | numpy_engine.NumpyEngine, ModuleType, bytes]:
    if backend == "faiss":
        index_path.write_bytes(b"old-faiss-index")
        engine = object.__new__(faiss_engine.FaissEngine)
        engine.index = object()
        engine._faiss = _FakeFaiss()
        return engine, faiss_engine, b"new-faiss-index"

    with index_path.open("wb") as target:
        np.save(target, np.asarray([[1.0, 0.0]], dtype="float32"), allow_pickle=False)
    engine = numpy_engine.NumpyEngine()
    engine.load_or_initialize(index_path, 2)
    engine.rebuild(np.asarray([[0.0, 1.0]], dtype="float32"))
    return engine, numpy_engine, b""


@pytest.mark.parametrize("backend", ["faiss", "numpy"])
def test_engine_save_replaces_complete_temp_file(tmp_path, monkeypatch, backend):
    index_path = tmp_path / "index.bin"
    engine, module, expected_bytes = _engine_case(backend, index_path)
    calls: list[tuple[Path, Path]] = []

    def replace(temp_path: Path, target_path: Path) -> None:
        assert temp_path.is_file()
        assert target_path == index_path
        calls.append((temp_path, target_path))
        atomic_replace_file(temp_path, target_path)

    monkeypatch.setattr(module, "atomic_replace_file", replace)
    engine.save(index_path)

    assert len(calls) == 1
    assert not list(tmp_path.glob("*.tmp"))
    if backend == "faiss":
        assert index_path.read_bytes() == expected_bytes
    else:
        with index_path.open("rb") as saved:
            np.testing.assert_array_equal(
                np.load(saved, allow_pickle=False),
                np.asarray([[0.0, 1.0]], dtype="float32"),
            )


@pytest.mark.parametrize("backend", ["faiss", "numpy"])
def test_engine_save_failure_keeps_prior_index_and_removes_temp(tmp_path, monkeypatch, backend):
    index_path = tmp_path / "index.bin"
    engine, module, _ = _engine_case(backend, index_path)
    original = index_path.read_bytes()

    def fail_before_replace(temp_path: Path, target_path: Path) -> None:
        assert temp_path.is_file()
        assert target_path == index_path
        raise OSError("injected sync failure")

    monkeypatch.setattr(module, "atomic_replace_file", fail_before_replace)
    with pytest.raises(OSError, match="injected sync failure"):
        engine.save(index_path)

    assert index_path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
