from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from local_rag_backend.infrastructure.persistence.faiss.index import _exclusive_file_lock


def test_exclusive_file_lock_fails_closed_when_os_locking_is_unavailable(tmp_path, monkeypatch):
    def _raise_lock_error(*_args, **_kwargs):
        raise OSError("locking unavailable")

    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(LOCK_EX=1, LOCK_UN=2, flock=_raise_lock_error),
    )
    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        SimpleNamespace(LK_LOCK=1, LK_UNLCK=0, locking=_raise_lock_error),
    )

    with (
        pytest.raises(RuntimeError, match="Unable to acquire FAISS file lock"),
        _exclusive_file_lock(tmp_path / "faiss.lock"),
    ):
        pass
