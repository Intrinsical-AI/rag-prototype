from __future__ import annotations

from types import SimpleNamespace

from local_rag_backend.core.use_cases.rag_query import (
    execute_ask_eval_sync,
    list_history_entries_sync,
)


def test_execute_ask_eval_sync_builds_and_runs_service() -> None:
    cfg = SimpleNamespace(k=2)

    class DummyRuntimeFactory:
        def run_ask_eval(self, *, question: str, cfg):
            assert question == "hello?"
            assert cfg.k == 2
            return {"answer": "ok", "docs": [], "scores": []}

    outcome = execute_ask_eval_sync(
        question="hello?",
        cfg=cfg,
        rag_runtime_factory=DummyRuntimeFactory(),
    )

    assert outcome.rag_result["answer"] == "ok"
    assert isinstance(outcome.latency_ms, int)
    assert outcome.latency_ms >= 0


def test_list_history_entries_sync_delegates_to_crud(monkeypatch) -> None:
    fake_rows = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    called: dict[str, int] = {}

    class DummyHistoryReader:
        def list_history_entries(self, *, limit: int, offset: int):
            called["limit"] = limit
            called["offset"] = offset
            return tuple(fake_rows)

    out = list_history_entries_sync(history_reader=DummyHistoryReader(), limit=3, offset=1)
    assert out == tuple(fake_rows)
    assert called == {"limit": 3, "offset": 1}
