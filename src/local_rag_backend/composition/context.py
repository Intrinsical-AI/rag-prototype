"""Shared application runtime context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from local_rag_backend.composition.runtime import RuntimeSnapshot, build_runtime_snapshot

if TYPE_CHECKING:
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


__all__ = ["AppContext"]
