from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from local_rag_backend.core.services import write_lock
from local_rag_backend.core.services.write_lock import _exclusive_file_lock


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
        pytest.raises(RuntimeError, match="Unable to acquire multi-store write lock"),
        _exclusive_file_lock(tmp_path / "write.lock"),
    ):
        pass


def test_multi_store_write_lock_uses_coordination_dir(tmp_path, monkeypatch):
    captured: list[object] = []

    @contextmanager
    def _fake_lock(path, *, timeout_s: float = 30.0, poll_s: float = 0.05):
        captured.append(path)
        yield

    monkeypatch.setattr(
        write_lock,
        "settings",
        SimpleNamespace(get_coordination_dir=lambda: tmp_path),
        raising=True,
    )
    monkeypatch.setattr(write_lock, "_exclusive_file_lock", _fake_lock, raising=True)

    with write_lock.multi_store_write_lock():
        pass

    assert captured == [tmp_path / ".rag_multi_store_write.lock"]
