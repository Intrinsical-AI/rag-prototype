"""Explicit test overrides for the default application's injectable factories."""

from __future__ import annotations

from typing import Any

import pytest

from local_rag_backend.composition import factory
from local_rag_backend.composition.context import AppContext


def override_container(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """Apply factory overrides to the current context and subsequent resets."""
    container = factory.get_app_context().container
    for name, value in overrides.items():
        monkeypatch.setattr(container, name, value)
    build_context = factory._build_app_context

    def build_overridden_context() -> AppContext:
        context = build_context()
        for name, value in overrides.items():
            setattr(context.container, name, value)
        return context

    monkeypatch.setattr(factory, "_build_app_context", build_overridden_context)
