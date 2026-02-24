"""Shared application runtime context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_rag_backend.app.container import AppContainer
    from local_rag_backend.settings import Settings


@dataclass(frozen=True, slots=True)
class AppContext:
    """Runtime context injected in HTTP dependencies."""

    settings_obj: Settings
    container: AppContainer

    @property
    def settings(self) -> Settings:
        return self.settings_obj


__all__ = ["AppContext"]
