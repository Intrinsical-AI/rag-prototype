"""Shared application runtime context."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

from local_rag_backend.composition.runtime import RuntimeSnapshot, build_runtime_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.composition.container import AppContainer
    from local_rag_backend.settings import Settings


@dataclass(frozen=True, slots=True)
class AppContext:
    """Runtime context injected in HTTP dependencies."""

    settings_obj: Settings
    container: AppContainer

    @property
    def settings(self) -> Settings:
        return self.settings_obj

    @property
    def runtime_snapshot(self) -> RuntimeSnapshot:
        """Derived runtime view for agent-facing status and transport surfaces."""
        return build_runtime_snapshot(self.settings_obj)


class ApplicationRuntime:
    """Own one application's lazy context and close its resources as one lifetime."""

    def __init__(self, context_factory: Callable[[], AppContext]) -> None:
        self._context_factory = context_factory
        self._context: AppContext | None = None
        self._lock = Lock()

    def context(self) -> AppContext:
        with self._lock:
            if self._context is None:
                self._context = self._context_factory()
            return self._context

    def close(self) -> None:
        with self._lock:
            if self._context is not None:
                self._context.container.close()
                self._context = None


__all__ = ["AppContext", "ApplicationRuntime"]
