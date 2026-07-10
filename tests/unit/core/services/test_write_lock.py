from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from local_rag_backend.infrastructure.concurrency.locks import write_lock
from local_rag_backend.infrastructure.concurrency.locks.write_lock import _exclusive_file_lock


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


def test_multi_store_write_lock_records_wait_and_hold_metrics(tmp_path, monkeypatch):
    metrics_path = tmp_path / "lock_metrics.ndjson"

    @contextmanager
    def _fake_lock(path, *, timeout_s: float = 30.0, poll_s: float = 0.05):
        _ = path, timeout_s, poll_s
        yield

    mono = iter([10.0, 10.2, 10.2, 10.7])
    monkeypatch.setattr(
        write_lock,
        "settings",
        SimpleNamespace(get_coordination_dir=lambda: tmp_path, lock_metrics_path=str(metrics_path)),
        raising=True,
    )
    monkeypatch.setattr(write_lock, "_exclusive_file_lock", _fake_lock, raising=True)
    monkeypatch.setattr(write_lock.time, "monotonic", lambda: next(mono), raising=True)
    monkeypatch.setattr(write_lock.time, "time", lambda: 123.0, raising=True)

    with write_lock.multi_store_write_lock():
        pass

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in rows] == ["acquired", "released"]
    assert rows[0]["wait_ms"] == pytest.approx(200.0, abs=0.001)
    assert rows[1]["hold_ms"] == pytest.approx(500.0, abs=0.001)


def test_multi_store_write_lock_records_released_and_failed_on_exception(tmp_path, monkeypatch):
    metrics_path = tmp_path / "lock_metrics_error.ndjson"

    @contextmanager
    def _fake_lock(path, *, timeout_s: float = 30.0, poll_s: float = 0.05):
        _ = path, timeout_s, poll_s
        yield

    mono = iter([1.0, 1.1, 1.1, 1.6, 1.8])
    monkeypatch.setattr(
        write_lock,
        "settings",
        SimpleNamespace(get_coordination_dir=lambda: tmp_path, lock_metrics_path=str(metrics_path)),
        raising=True,
    )
    monkeypatch.setattr(write_lock, "_exclusive_file_lock", _fake_lock, raising=True)
    monkeypatch.setattr(write_lock.time, "monotonic", lambda: next(mono), raising=True)
    monkeypatch.setattr(write_lock.time, "time", lambda: 321.0, raising=True)

    with pytest.raises(RuntimeError, match="boom"), write_lock.multi_store_write_lock():
        raise RuntimeError("boom")

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in rows] == ["acquired", "released", "failed"]
    assert rows[1]["hold_ms"] == pytest.approx(500.0, abs=0.001)
    assert rows[2]["wait_ms"] == pytest.approx(800.0, abs=0.001)
