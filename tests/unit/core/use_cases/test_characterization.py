from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_rag_backend.core.errors import LLMResponseError
from local_rag_backend.core.use_cases import docs_query, evaluation, health, mutations, openrouter
from local_rag_backend.core.use_cases.rag_query import list_history_entries_sync


def test_docs_query_list_docs_page_sync_delegates_to_port() -> None:
    captured: dict[str, int] = {}
    expected = (
        SimpleNamespace(id="doc:1", content="hello"),
        SimpleNamespace(id="doc:2", content="world"),
    )

    class DummyDocsReadPort:
        def list_docs_page(self, *, limit: int, offset: int):
            captured["limit"] = limit
            captured["offset"] = offset
            return expected

    out = docs_query.list_docs_page_sync(docs_reader=DummyDocsReadPort(), limit=10, offset=3)
    assert out == expected
    assert captured == {"limit": 10, "offset": 3}


def test_health_check_database_sets_failed_status_on_ping_error(monkeypatch) -> None:
    checks: dict[str, object] = {}

    class FailingDiagnostics:
        def ping_database(self):
            raise RuntimeError("db")

    is_ok = health.check_database(checks=checks, diagnostics=FailingDiagnostics())
    assert is_ok is False
    assert str(checks["database"]).startswith("failed:")


def test_openrouter_generate_openrouter_sync_wraps_port_errors() -> None:
    class FailingClient:
        def generate(self, *, request):
            raise RuntimeError("boom")

    payload = openrouter.OpenRouterGenerateInput(
        model=None,
        system_instruction="sys",
        user_content="hi",
        temperature=0.2,
        max_tokens=10,
        top_p=0.9,
    )
    with pytest.raises(LLMResponseError, match="OpenRouter error: boom"):
        openrouter.generate_openrouter_sync(payload=payload, openrouter_client=FailingClient())


async def test_mutations_run_api_mutation_executes_with_blocking_executor() -> None:
    seen: dict[str, str] = {}

    class DummyBlockingExecutor:
        async def run_blocking(self, func, /, *args, **kwargs):
            seen["task_type"] = str(kwargs.get("task_type", "default"))
            return func(*args)

    out = await mutations.run_api_mutation(
        operation=lambda: "ok",
        run_locked=lambda fn: fn(),
        reset_after=lambda: None,
        blocking_executor=DummyBlockingExecutor(),
    )
    assert out == "ok"
    assert seen["task_type"] == "mutation"


def test_rag_query_list_history_entries_sync_delegates_to_crud() -> None:
    rows = [SimpleNamespace(id=1)]

    class DummyHistoryReader:
        def list_history_entries(self, *, limit: int, offset: int):
            assert limit == 5
            assert offset == 2
            return tuple(rows)

    assert list_history_entries_sync(
        history_reader=DummyHistoryReader(), limit=5, offset=2
    ) == tuple(rows)


def test_evaluation_run_retrieval_eval_validates_inputs() -> None:
    dataset = SimpleNamespace(docs=[], dataset_id="d", queries=[])

    with pytest.raises(ValueError, match="retrieval_mode=sparse only"):
        evaluation.run_retrieval_eval(
            dataset=dataset,
            eval_storage_port=SimpleNamespace(),
            eval_retriever_factory_port=SimpleNamespace(),
            retrieval_mode="dense",
        )

    with pytest.raises(ValueError, match="k must be positive"):
        evaluation.run_retrieval_eval(
            dataset=dataset,
            eval_storage_port=SimpleNamespace(),
            eval_retriever_factory_port=SimpleNamespace(),
            retrieval_mode="sparse",
            k=0,
        )
