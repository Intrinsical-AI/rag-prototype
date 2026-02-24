"""
Utilities for running blocking (sync) code from async request handlers.

Important: Starlette/FastAPI commonly use AnyIO for their own threadpool helpers, but
this project is intentionally lightweight and runs under plain asyncio in tests.
Using a dedicated ThreadPoolExecutor avoids relying on AnyIO's context detection.
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from local_rag_backend.app.telemetry import get_telemetry

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
BlockingTaskType = Literal["default", "mutation", "network", "eval"]

_TASK_TYPES: tuple[BlockingTaskType, ...] = ("default", "mutation", "network", "eval")
_DEFAULT_WORKERS_BY_TASK: dict[BlockingTaskType, int] = {
    "default": 8,
    "mutation": 2,
    "network": 4,
    "eval": 2,
}
_DEFAULT_QUEUE_LIMIT_BY_TASK: dict[BlockingTaskType, int] = {
    "default": 64,
    "mutation": 32,
    "network": 64,
    "eval": 32,
}

_EXECUTOR_STATES: dict[BlockingTaskType, _ExecutorState] = {}
_EXECUTOR_LOCK = threading.Lock()
_POLL_INTERVAL_SECONDS = 0.001


def _parse_positive_int(raw: str | None, *, fallback: int) -> int:
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return value if value > 0 else fallback


def _default_workers() -> int:
    # Keep conservative by default; these tasks can involve network I/O and CPU.
    return _parse_positive_int(
        os.getenv("RAG_BLOCKING_WORKERS"), fallback=_DEFAULT_WORKERS_BY_TASK["default"]
    )


def _workers_for_task(task_type: BlockingTaskType) -> int:
    env_name = f"RAG_BLOCKING_WORKERS_{task_type.upper()}"
    raw = os.getenv(env_name)
    if raw is not None:
        return _parse_positive_int(raw, fallback=_DEFAULT_WORKERS_BY_TASK[task_type])
    if task_type == "default":
        return _default_workers()
    return _DEFAULT_WORKERS_BY_TASK[task_type]


def _queue_limit_for_task(task_type: BlockingTaskType) -> int:
    env_name = f"RAG_BLOCKING_QUEUE_{task_type.upper()}"
    return _parse_positive_int(
        os.getenv(env_name), fallback=_DEFAULT_QUEUE_LIMIT_BY_TASK[task_type]
    )


def _parse_task_type(task_type: str) -> BlockingTaskType:
    candidate = str(task_type).strip().lower()
    if candidate not in _TASK_TYPES:
        allowed = ", ".join(_TASK_TYPES)
        raise ValueError(
            f"Unsupported blocking task_type={task_type!r}. Expected one of: {allowed}"
        )
    return cast("BlockingTaskType", candidate)


@dataclass
class _ExecutorState:
    executor: ThreadPoolExecutor
    max_pending: int
    pending: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def try_acquire_slot(self) -> bool:
        with self.lock:
            if self.pending >= self.max_pending:
                return False
            self.pending += 1
            return True

    def release_slot(self) -> None:
        with self.lock:
            if self.pending > 0:
                self.pending -= 1

    def pending_snapshot(self) -> int:
        with self.lock:
            return int(self.pending)


def _get_executor_state(task_type: BlockingTaskType) -> _ExecutorState:
    parsed = _parse_task_type(task_type)
    with _EXECUTOR_LOCK:
        state = _EXECUTOR_STATES.get(parsed)
        if state is None:
            workers = _workers_for_task(parsed)
            queue_limit = _queue_limit_for_task(parsed)
            state = _ExecutorState(
                executor=ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix=f"rag-blocking-{parsed}",
                ),
                max_pending=max(1, workers + queue_limit),
            )
            _EXECUTOR_STATES[parsed] = state
        return state


@atexit.register
def _shutdown_executor() -> None:  # pragma: no cover
    for state in list(_EXECUTOR_STATES.values()):
        # Best-effort shutdown; don't block interpreter exit.
        state.executor.shutdown(wait=False, cancel_futures=True)


async def run_blocking(
    func: Callable[..., T],
    /,
    *args: Any,
    task_type: BlockingTaskType = "default",
    **kwargs: Any,
) -> T:
    """Run a sync callable in a dedicated worker pool partitioned by task type."""
    telemetry = get_telemetry()
    call = functools.partial(func, *args, **kwargs)
    state = _get_executor_state(task_type)
    capacity = int(state.max_pending)
    if not state.try_acquire_slot():
        telemetry.observe_blocking_queue(
            task_type=task_type,
            pending=_pending_snapshot(state),
            capacity=capacity,
        )
        telemetry.observe_blocking_run(task_type=task_type, status="rejected", duration_s=0.0)
        raise RuntimeError(
            f"Blocking queue is full for task_type={task_type!r} (max_pending={state.max_pending})"
        )
    telemetry.observe_blocking_queue(
        task_type=task_type,
        pending=_pending_snapshot(state),
        capacity=capacity,
    )

    # Note: `loop.run_in_executor()` / `asyncio.to_thread()` / `asyncio.wrap_future()` rely on
    # cross-thread wakeups (`loop.call_soon_threadsafe()`), which can deadlock under some
    # ASGI test harnesses. Polling avoids that class of deadlocks at the cost of a tiny
    # timer wakeup while the job runs.
    enqueued_at = time.monotonic()
    started_at = enqueued_at

    def _instrumented_call() -> T:
        nonlocal started_at
        started_at = time.monotonic()
        telemetry.observe_blocking_queue_wait(
            task_type=task_type,
            wait_s=max(0.0, started_at - enqueued_at),
        )
        return call()

    fut = None
    status: Literal["ok", "error", "cancelled"] = "ok"
    try:
        fut = state.executor.submit(_instrumented_call)
        while not fut.done():
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        return fut.result()
    except asyncio.CancelledError:  # pragma: no cover
        status = "cancelled"
        if fut is not None:
            fut.cancel()
        raise
    except Exception:
        status = "error"
        raise
    finally:
        telemetry.observe_blocking_run(
            task_type=task_type,
            status=status,
            duration_s=max(0.0, time.monotonic() - started_at),
        )
        state.release_slot()
        telemetry.observe_blocking_queue(
            task_type=task_type,
            pending=_pending_snapshot(state),
            capacity=capacity,
        )


def _pending_snapshot(state: _ExecutorState) -> int:
    if hasattr(state, "pending_snapshot"):
        return int(state.pending_snapshot())
    pending = getattr(state, "pending", 0)
    return int(pending) if isinstance(pending, int) else 0
