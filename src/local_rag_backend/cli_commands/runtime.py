"""Shared runtime helpers for CLI command modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from local_rag_backend.composition.adapters import build_dense_embedder_from_settings
from local_rag_backend.settings import settings

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
    Ensure SQLite schema is compatible with the current ORM mappings.

    CLI commands can be run without starting the FastAPI server, so they must
    apply the same best-effort SQLite migrations that the app does at startup.
    """
    from local_rag_backend.infrastructure.persistence.sql import base as db_base

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db_base.ensure_sqlite_schema_compatible(
        engine_to_use=db_base.engine, id_map_path=str(settings.id_map_path)
    )


def _run_with_multi_store_write_lock(operation: Callable[[], T]) -> T:
    from local_rag_backend.infrastructure.concurrency.locks.write_lock import (
        multi_store_write_lock,
    )

    with multi_store_write_lock():
        return operation()


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
    return build_dense_embedder_from_settings(settings_obj=settings)


__all__ = [
    "build_dense_embedder",
    "ensure_sqlite_schema_for_cli",
    "get_cli_container",
    "run_cli_mutation",
]
