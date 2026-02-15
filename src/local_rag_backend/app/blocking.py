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
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _default_workers() -> int:
    # Keep conservative by default; these tasks can involve network I/O and CPU.
    env = os.getenv("RAG_BLOCKING_WORKERS")
    if env:
        try:
            v = int(env)
        except ValueError:
            return 8
        if v > 0:
            return v
    return 8


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=_default_workers(),
                thread_name_prefix="rag-blocking",
            )
    return _EXECUTOR


@atexit.register
def _shutdown_executor() -> None:  # pragma: no cover
    ex = _EXECUTOR
    if ex is not None:
        # Best-effort shutdown; don't block interpreter exit.
        ex.shutdown(wait=False, cancel_futures=True)


async def run_blocking(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a sync callable in a dedicated worker pool."""
    call = functools.partial(func, *args, **kwargs)
    # Note: `loop.run_in_executor()` / `asyncio.to_thread()` / `asyncio.wrap_future()` rely on
    # cross-thread wakeups (`loop.call_soon_threadsafe()`), which can deadlock under some
    # ASGI test harnesses. Polling avoids that class of deadlocks at the cost of a tiny
    # timer wakeup while the job runs.
    fut = _get_executor().submit(call)
    try:
        while not fut.done():
            await asyncio.sleep(0.001)
        return fut.result()
    except asyncio.CancelledError:  # pragma: no cover
        fut.cancel()
        raise
