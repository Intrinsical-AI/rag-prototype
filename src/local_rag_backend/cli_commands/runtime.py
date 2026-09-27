"""Shared runtime helpers for CLI command modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from local_rag_backend.composition.runtime import RuntimeSnapshot, build_runtime_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.composition.container import AppContainer
    from local_rag_backend.core.ports import EmbedderPort

T = TypeVar("T")


def _noop_ensure() -> None:
    return None


def _run_without_lock(fn: Callable[[], T]) -> T:
    return fn()


def ensure_sqlite_schema_for_cli() -> None:
    """
    Ensure the current SQLite ORM schema exists.

    CLI commands can be run without starting the FastAPI server, so they must
    apply the same fresh-schema bootstrap that the app does at startup.
    """
    get_cli_container().initialize()


def _run_with_multi_store_write_lock(operation: Callable[[], T]) -> T:
    return cast(T, get_cli_container().run_multi_store_write_locked(operation))


def _reset_rag_service_best_effort() -> None:
    from local_rag_backend.composition.factory import reset_rag_service

    try:
        reset_rag_service()
    except Exception:
        return None


def get_cli_container() -> AppContainer:
    """Resolve the app container for CLI wiring."""
    from local_rag_backend.composition.factory import get_app_context

    return get_app_context().container


def get_cli_runtime_snapshot() -> RuntimeSnapshot:
    """Return the current CLI runtime snapshot without exposing raw Settings."""
    return build_runtime_snapshot(get_cli_container().settings_obj)


def run_cli_mutation(
    operation: Callable[[], T],
    *,
    use_lock: bool = True,
    ensure_schema: bool = True,
) -> T:
    from local_rag_backend.core.use_cases.mutations import (
        run_cli_mutation as run_cli_mutation_core,
    )

    ensure_fn: Callable[[], None] = ensure_sqlite_schema_for_cli if ensure_schema else _noop_ensure
    run_locked_fn = cast(
        "Callable[[Callable[[], T]], T]",
        _run_with_multi_store_write_lock if use_lock else _run_without_lock,
    )

    return run_cli_mutation_core(
        operation=operation,
        ensure_schema=ensure_fn,
        run_locked=run_locked_fn,
        reset_after=_reset_rag_service_best_effort,
    )


def build_dense_embedder() -> EmbedderPort:
    """Build the dense/hybrid embedder based on current settings."""
    return get_cli_container().build_dense_embedder()


__all__ = [
    "build_dense_embedder",
    "ensure_sqlite_schema_for_cli",
    "get_cli_container",
    "get_cli_runtime_snapshot",
    "run_cli_mutation",
]
