from __future__ import annotations

import pytest

from local_rag_backend.app import blocking


def test_default_workers_uses_valid_positive_env(monkeypatch):
    monkeypatch.setenv("RAG_BLOCKING_WORKERS", "12")
    assert blocking._default_workers() == 12


def test_default_workers_falls_back_on_invalid_or_non_positive_env(monkeypatch):
    monkeypatch.setenv("RAG_BLOCKING_WORKERS", "abc")
    assert blocking._default_workers() == 8

    monkeypatch.setenv("RAG_BLOCKING_WORKERS", "0")
    assert blocking._default_workers() == 8

    monkeypatch.setenv("RAG_BLOCKING_WORKERS", "-3")
    assert blocking._default_workers() == 8


@pytest.mark.unit
async def test_run_blocking_returns_result():
    out = await blocking.run_blocking(lambda x, y: x + y, 2, 3)
    assert out == 5


@pytest.mark.unit
async def test_run_blocking_propagates_sync_exception():
    def _boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await blocking.run_blocking(_boom)
