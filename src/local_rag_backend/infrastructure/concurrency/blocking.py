"""
Utilities for running blocking (sync) code from async request handlers.

Important: Starlette/FastAPI commonly use AnyIO for their own threadpool helpers, but
this project is intentionally lightweight and runs under plain asyncio in tests.
Using a dedicated ThreadPoolExecutor avoids relying on AnyIO's context detection.
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from local_rag_backend.infrastructure.observability.telemetry import get_telemetry
from local_rag_backend.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

T = TypeVar("T")
BlockingTaskType = Literal["default", "mutation", "network", "eval"]

_TASK_TYPES: tuple[BlockingTaskType, ...] = ("default", "mutation", "network", "eval")
_POLL_INTERVAL_SECONDS = 0.001


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


class BlockingExecutor:
    """Bounded worker pools owned by one application container."""

    def __init__(self, *, settings_obj: Settings) -> None:
        self._settings = settings_obj
        self._states: dict[BlockingTaskType, _ExecutorState] = {}
        self._lock = threading.Lock()
        self._closed = False

    def _get_executor_state(self, task_type: BlockingTaskType) -> _ExecutorState:
        parsed = _parse_task_type(task_type)
        with self._lock:
            if self._closed:
                raise RuntimeError("The application worker executor is closed.")
            state = self._states.get(parsed)
            if state is None:
                suffix = "" if parsed == "default" else f"_{parsed}"
                workers = int(getattr(self._settings, f"blocking_workers{suffix}"))
                queue_limit = int(getattr(self._settings, f"blocking_queue_{parsed}"))
                state = _ExecutorState(
                    executor=ThreadPoolExecutor(
                        max_workers=workers, thread_name_prefix=f"rag-blocking-{parsed}"
                    ),
                    max_pending=workers + queue_limit,
                )
                self._states[parsed] = state
            return state

    def close(self) -> None:
        with self._lock:
            self._closed = True
            states = tuple(self._states.values())
            self._states.clear()
        for state in states:
            state.executor.shutdown(wait=True, cancel_futures=True)

    async def run_blocking(
        self,
        func: Callable[..., T],
        /,
        *args: Any,
        task_type: BlockingTaskType = "default",
        **kwargs: Any,
    ) -> T:
        """Run a sync callable in a dedicated worker pool partitioned by task type."""
        telemetry = get_telemetry()
        call = functools.partial(func, *args, **kwargs)
        state = self._get_executor_state(task_type)
        capacity = int(state.max_pending)
        if not state.try_acquire_slot():
            telemetry.observe_blocking_queue(
                task_type=task_type,
                pending=state.pending_snapshot(),
                capacity=capacity,
            )
            telemetry.observe_blocking_run(task_type=task_type, status="rejected", duration_s=0.0)
            raise RuntimeError(
                f"Blocking queue is full for task_type={task_type!r} (max_pending={state.max_pending})"
            )
        telemetry.observe_blocking_queue(
            task_type=task_type,
            pending=state.pending_snapshot(),
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

        def _release_completed_task(_future: Future[T]) -> None:
            state.release_slot()
            telemetry.observe_blocking_queue(
                task_type=task_type,
                pending=state.pending_snapshot(),
                capacity=capacity,
            )

        try:
            fut = state.executor.submit(_instrumented_call)
        except RuntimeError:
            state.release_slot()
            raise
        fut.add_done_callback(_release_completed_task)
        status: Literal["ok", "error", "cancelled"] = "ok"
        try:
            while not fut.done():
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            return fut.result()
        except asyncio.CancelledError:
            status = "cancelled"
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
