from __future__ import annotations

from concurrent.futures import Future

import pytest

from local_rag_backend.app import blocking


@pytest.fixture(autouse=True)
def _reset_blocking_state() -> None:
    blocking._shutdown_executor()
    blocking._EXECUTOR_STATES.clear()
    yield
    blocking._shutdown_executor()
    blocking._EXECUTOR_STATES.clear()


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


def test_workers_for_task_prefers_task_specific_env(monkeypatch):
    monkeypatch.setenv("RAG_BLOCKING_WORKERS_MUTATION", "5")
    assert blocking._workers_for_task("mutation") == 5


def test_executor_state_uses_workers_and_queue_limits(monkeypatch):
    monkeypatch.setenv("RAG_BLOCKING_WORKERS_MUTATION", "3")
    monkeypatch.setenv("RAG_BLOCKING_QUEUE_MUTATION", "9")

    state = blocking._get_executor_state("mutation")
    assert state.max_pending == 12


@pytest.mark.unit
async def test_run_blocking_returns_result():
    out = await blocking.run_blocking(lambda x, y: x + y, 2, 3)
    assert out == 5


@pytest.mark.unit
async def test_run_blocking_routes_task_type_and_releases_slot(monkeypatch):
    seen_task_types: list[str] = []

    class _FakeExecutor:
        def submit(self, call):
            fut: Future[str] = Future()
            fut.set_result(call())
            return fut

    class _FakeState:
        def __init__(self) -> None:
            self.executor = _FakeExecutor()
            self.max_pending = 10
            self.acquired = 0
            self.released = 0

        def try_acquire_slot(self) -> bool:
            self.acquired += 1
            return True

        def release_slot(self) -> None:
            self.released += 1

    state = _FakeState()

    def _fake_get_executor_state(task_type):
        seen_task_types.append(task_type)
        return state

    monkeypatch.setattr(blocking, "_get_executor_state", _fake_get_executor_state, raising=True)

    out = await blocking.run_blocking(lambda: "ok", task_type="mutation")
    assert out == "ok"
    assert seen_task_types == ["mutation"]
    assert state.acquired == 1
    assert state.released == 1


@pytest.mark.unit
async def test_run_blocking_raises_when_queue_full(monkeypatch):
    class _FakeTelemetry:
        def __init__(self) -> None:
            self.queue: list[tuple[str, int, int]] = []
            self.runs: list[tuple[str, str]] = []

        def observe_blocking_queue(self, *, task_type, pending, capacity) -> None:
            self.queue.append((str(task_type), int(pending), int(capacity)))

        def observe_blocking_run(self, *, task_type, status, duration_s) -> None:
            self.runs.append((str(task_type), str(status)))

        def observe_blocking_queue_wait(self, *, task_type, wait_s) -> None:
            return None

    telemetry = _FakeTelemetry()
    monkeypatch.setattr(blocking, "get_telemetry", lambda: telemetry, raising=True)

    class _FullState:
        max_pending = 1

        class _Executor:
            def submit(self, _call):
                raise AssertionError("submit must not be called when queue is full")

        executor = _Executor()

        def try_acquire_slot(self) -> bool:
            return False

        def release_slot(self) -> None:
            raise AssertionError("release_slot must not be called when slot was not acquired")

    monkeypatch.setattr(blocking, "_get_executor_state", lambda _task: _FullState(), raising=True)

    with pytest.raises(RuntimeError, match="queue is full"):
        await blocking.run_blocking(lambda: 1, task_type="eval")

    assert telemetry.queue == [("eval", 0, 1)]
    assert telemetry.runs == [("eval", "rejected")]


@pytest.mark.unit
async def test_run_blocking_releases_slot_when_sync_callable_raises(monkeypatch):
    class _FakeExecutor:
        def submit(self, call):
            fut: Future[object] = Future()
            try:
                call()
            except Exception as e:  # pragma: no branch
                fut.set_exception(e)
            else:  # pragma: no cover
                fut.set_result(None)
            return fut

    class _FakeState:
        def __init__(self) -> None:
            self.executor = _FakeExecutor()
            self.max_pending = 10
            self.released = 0

        def try_acquire_slot(self) -> bool:
            return True

        def release_slot(self) -> None:
            self.released += 1

    state = _FakeState()
    monkeypatch.setattr(blocking, "_get_executor_state", lambda _task: state, raising=True)

    def _boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await blocking.run_blocking(_boom, task_type="network")

    assert state.released == 1


@pytest.mark.unit
async def test_run_blocking_reports_queue_saturation_and_wait(monkeypatch):
    class _FakeTelemetry:
        def __init__(self) -> None:
            self.queue: list[tuple[str, int, int]] = []
            self.waits: list[tuple[str, float]] = []
            self.runs: list[tuple[str, str]] = []

        def observe_blocking_queue(self, *, task_type, pending, capacity) -> None:
            self.queue.append((str(task_type), int(pending), int(capacity)))

        def observe_blocking_queue_wait(self, *, task_type, wait_s) -> None:
            self.waits.append((str(task_type), float(wait_s)))

        def observe_blocking_run(self, *, task_type, status, duration_s) -> None:
            self.runs.append((str(task_type), str(status)))

    telemetry = _FakeTelemetry()
    monkeypatch.setattr(blocking, "get_telemetry", lambda: telemetry, raising=True)

    class _FakeExecutor:
        def submit(self, call):
            fut: Future[str] = Future()
            fut.set_result(call())
            return fut

    class _FakeState:
        def __init__(self) -> None:
            self.executor = _FakeExecutor()
            self.max_pending = 5
            self.pending = 0

        def try_acquire_slot(self) -> bool:
            self.pending += 1
            return True

        def release_slot(self) -> None:
            self.pending -= 1

    state = _FakeState()
    monkeypatch.setattr(blocking, "_get_executor_state", lambda _task: state, raising=True)

    out = await blocking.run_blocking(lambda: "ok", task_type="mutation")
    assert out == "ok"
    assert telemetry.queue[0] == ("mutation", 1, 5)
    assert telemetry.queue[-1] == ("mutation", 0, 5)
    assert telemetry.waits and telemetry.waits[0][0] == "mutation"
    assert telemetry.runs == [("mutation", "ok")]


@pytest.mark.unit
async def test_run_blocking_rejects_unknown_task_type():
    with pytest.raises(ValueError, match="Unsupported blocking task_type"):
        await blocking.run_blocking(lambda: 1, task_type="invalid")  # type: ignore[arg-type]
