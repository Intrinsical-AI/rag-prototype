"""Shared mutation executor for API and CLI entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from local_rag_backend.core.use_cases.errors import map_runtime_error
from local_rag_backend.infrastructure.concurrency.blocking import run_blocking

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from local_rag_backend.core.use_cases.errors import AppError
    from local_rag_backend.infrastructure.concurrency.blocking import BlockingTaskType

T = TypeVar("T")


def _map_exception(
    exc: Exception,
    *,
    map_error: Callable[[Exception], AppError | None] | None,
) -> Exception:
    if map_error is not None:
        mapped = map_error(exc)
        if mapped is not None:
            return mapped
    mapped_runtime = map_runtime_error(exc)
    if mapped_runtime is not None:
        return mapped_runtime
    return exc


async def run_api_mutation(
    *,
    operation: Callable[[], T],
    run_locked: Callable[[Callable[[], T]], T],
    reset_after: Callable[[], None],
    run_blocking_fn: Callable[..., Awaitable[T]] = run_blocking,
    task_type: BlockingTaskType = "mutation",
    map_error: Callable[[Exception], AppError | None] | None = None,
) -> T:
    """Run one HTTP mutation with lock + offload + reset + mapped errors."""
    try:
        return await run_blocking_fn(run_locked, operation, task_type=task_type)
    except Exception as exc:
        mapped = _map_exception(exc, map_error=map_error)
        if mapped is not exc:
            raise mapped from exc
        raise
    finally:
        reset_after()


def run_cli_mutation(
    *,
    operation: Callable[[], T],
    ensure_schema: Callable[[], None],
    run_locked: Callable[[Callable[[], T]], T],
    reset_after: Callable[[], None],
    map_error: Callable[[Exception], AppError | None] | None = None,
) -> T:
    """Run one CLI mutation with schema ensure + lock + reset + mapped errors."""
    ensure_schema()
    try:
        return run_locked(operation)
    except Exception as exc:
        mapped = _map_exception(exc, map_error=map_error)
        if mapped is not exc:
            raise mapped from exc
        raise
    finally:
        reset_after()
