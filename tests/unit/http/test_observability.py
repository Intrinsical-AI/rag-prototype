from __future__ import annotations

from local_rag_backend.infrastructure.observability import observability


def test_observability_delegates_to_telemetry(monkeypatch) -> None:
    class _FakeTelemetry:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []
            self.queries: list[tuple[str, bool, bool]] = []
            self.ingests: list[tuple[str, bool, int]] = []

        def log_event(self, event: str, **fields: object) -> None:
            self.events.append((event, dict(fields)))

        def observe_query(
            self,
            *,
            retrieval_mode: str,
            reranker_enabled: bool,
            ok: bool,
            duration_s: float,
        ) -> None:
            self.queries.append((retrieval_mode, reranker_enabled, ok))

        def observe_ingest(self, *, source: str, ok: bool, inserted: int) -> None:
            self.ingests.append((source, ok, inserted))

        def observe_blocking_queue(self, *, task_type, pending, capacity) -> None:
            return None

        def observe_blocking_queue_wait(self, *, task_type, wait_s) -> None:
            return None

        def observe_blocking_run(self, *, task_type, status, duration_s) -> None:
            return None

    sink = _FakeTelemetry()
    monkeypatch.setattr(observability, "get_telemetry", lambda: sink, raising=True)

    observability.log_event("e", a=1)
    observability.observe_query(
        retrieval_mode="sparse",
        reranker_enabled=False,
        ok=True,
        duration_s=0.01,
    )
    observability.observe_ingest(source="api:/docs/ingest", ok=True, inserted=2)

    assert sink.events == [("e", {"a": 1})]
    assert sink.queries == [("sparse", False, True)]
    assert sink.ingests == [("api:/docs/ingest", True, 2)]
