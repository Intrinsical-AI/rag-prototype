from __future__ import annotations

import pytest

from local_rag_backend.core.errors import LLMTimeoutError
from local_rag_backend.core.use_cases.errors import BadRequestError, GatewayTimeoutError
from local_rag_backend.core.use_cases.mutations import run_api_mutation, run_cli_mutation


async def test_run_api_mutation_executes_locked_and_resets() -> None:
    reset_calls = 0
    blocking_calls: list[tuple[object, str]] = []

    def _operation() -> int:
        return 7

    def _run_locked(fn):
        return fn()

    def _reset() -> None:
        nonlocal reset_calls
        reset_calls += 1

    async def _fake_run_blocking(func, /, *args, **kwargs):
        task_type = str(kwargs.get("task_type", "default"))
        blocking_calls.append((func, task_type))
        return func(*args)

    out = await run_api_mutation(
        operation=_operation,
        run_locked=_run_locked,
        reset_after=_reset,
        run_blocking_fn=_fake_run_blocking,
    )
    assert out == 7
    assert len(blocking_calls) == 1
    assert blocking_calls[0][1] == "mutation"
    assert reset_calls == 1


async def test_run_api_mutation_maps_custom_error_and_resets() -> None:
    reset_calls = 0

    def _operation() -> int:
        raise ValueError("bad input")

    def _run_locked(fn):
        return fn()

    def _reset() -> None:
        nonlocal reset_calls
        reset_calls += 1

    async def _fake_run_blocking(func, /, *args, **kwargs):
        return func(*args)

    def _map_error(exc: Exception):
        if isinstance(exc, ValueError):
            return BadRequestError("mapped-bad-request")
        return None

    with pytest.raises(BadRequestError, match="mapped-bad-request"):
        await run_api_mutation(
            operation=_operation,
            run_locked=_run_locked,
            reset_after=_reset,
            run_blocking_fn=_fake_run_blocking,
            map_error=_map_error,
        )
    assert reset_calls == 1


def test_run_cli_mutation_ensures_schema_then_locks_then_resets() -> None:
    calls: list[str] = []

    def _ensure() -> None:
        calls.append("ensure")

    def _operation() -> int:
        calls.append("operation")
        return 3

    def _run_locked(fn):
        calls.append("lock")
        return fn()

    def _reset() -> None:
        calls.append("reset")

    out = run_cli_mutation(
        operation=_operation,
        ensure_schema=_ensure,
        run_locked=_run_locked,
        reset_after=_reset,
    )
    assert out == 3
    assert calls == ["ensure", "lock", "operation", "reset"]


def test_run_cli_mutation_maps_runtime_error() -> None:
    def _ensure() -> None:
        return None

    def _operation() -> int:
        raise LLMTimeoutError("provider timeout")

    def _run_locked(fn):
        return fn()

    def _reset() -> None:
        return None

    with pytest.raises(GatewayTimeoutError, match="provider timeout"):
        run_cli_mutation(
            operation=_operation,
            ensure_schema=_ensure,
            run_locked=_run_locked,
            reset_after=_reset,
        )
