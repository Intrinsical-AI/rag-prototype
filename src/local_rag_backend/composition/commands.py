"""Application composition for operational commands across CLI and MCP."""

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


def ensure_store() -> None:
    """
    Ensure the current SQLite ORM schema exists.

    CLI commands can be run without starting the FastAPI server, so they must
    apply the same fresh-schema bootstrap that the app does at startup.
    """
    get_container().initialize()


def _run_with_multi_store_write_lock(operation: Callable[[], T]) -> T:
    return cast(T, get_container().run_multi_store_write_locked(operation))


def get_container() -> AppContainer:
    """Resolve the app container for CLI wiring."""
    from local_rag_backend.composition.factory import get_app_context

    return get_app_context().container


def get_runtime_snapshot() -> RuntimeSnapshot:
    """Return the current CLI runtime snapshot without exposing raw Settings."""
    return build_runtime_snapshot(get_container().settings_obj)


def execute_mutation_command(
    operation: Callable[[], T],
    *,
    use_lock: bool = True,
    ensure_schema: bool = True,
) -> T:
    from local_rag_backend.core.use_cases.mutations import (
        execute_mutation_command as execute_mutation_command_core,
    )

    ensure_fn: Callable[[], None] = ensure_store if ensure_schema else _noop_ensure
    run_locked_fn = cast(
        "Callable[[Callable[[], T]], T]",
        _run_with_multi_store_write_lock if use_lock else _run_without_lock,
    )

    return execute_mutation_command_core(
        operation=operation,
        ensure_schema=ensure_fn,
        run_locked=run_locked_fn,
        reset_after=get_container().clear_local_rag_service_cache,
    )


def build_dense_embedder() -> EmbedderPort:
    """Build the dense/hybrid embedder based on current settings."""
    return get_container().build_dense_embedder()


__all__ = [
    "build_dense_embedder",
    "ensure_store",
    "execute_mutation_command",
    "get_container",
    "get_runtime_snapshot",
]
