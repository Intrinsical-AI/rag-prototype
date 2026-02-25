from __future__ import annotations

from types import SimpleNamespace

from local_rag_backend.core.use_cases.rag_query import (
    execute_ask_eval_sync,
    list_history_entries_sync,
)


def test_execute_ask_eval_sync_builds_and_runs_service() -> None:
    cfg = SimpleNamespace(k=2)

    class DummyRepo:
        def get_all_documents(self):
            return [SimpleNamespace(id=1, content="doc-1")]

    class DummyHistory:
        pass

    class DummyService:
        def __init__(self, retriever, generator, history_storage):
            self.retriever = retriever
            self.generator = generator
            self.history_storage = history_storage

        def ask(self, *, question: str, top_k: int):
            assert question == "hello?"
            assert top_k == 2
            return {"answer": "ok", "docs": [], "scores": []}

    outcome = execute_ask_eval_sync(
        question="hello?",
        cfg=cfg,
        build_retriever_from_config=lambda _cfg, _repo, **_k: "retriever",
        build_generator_from_config=lambda _cfg: "generator",
        doc_repo_factory=DummyRepo,
        history_repo_factory=DummyHistory,
        rag_service_factory=DummyService,
    )

    assert outcome.rag_result["answer"] == "ok"
    assert isinstance(outcome.latency_ms, int)
    assert outcome.latency_ms >= 0


def test_list_history_entries_sync_delegates_to_crud(monkeypatch) -> None:
    fake_rows = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    called: dict[str, int] = {}

    def _fake_get_history(*, db, limit: int, offset: int):
        assert db == "db"
        called["limit"] = limit
        called["offset"] = offset
        return fake_rows

    monkeypatch.setattr(
        "local_rag_backend.core.use_cases.rag_query.get_history",
        _fake_get_history,
        raising=True,
    )
    out = list_history_entries_sync(db="db", limit=3, offset=1)
    assert out == fake_rows
    assert called == {"limit": 3, "offset": 1}
