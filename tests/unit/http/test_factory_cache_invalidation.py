from __future__ import annotations

import pytest

from local_rag_backend.composition import factory
from local_rag_backend.infrastructure.persistence.sql import SystemStateStorage


@pytest.mark.unit
async def test_get_rag_service_cache_is_invalidated_by_system_state_version(
    in_memory_sqlite, monkeypatch
):
    """
    Regression test: cache invalidation must propagate across processes/workers.

    We simulate an external process by bumping the shared DB-backed version directly.
    """
    state = SystemStateStorage(session_factory=in_memory_sqlite)
    factory.reset_app_context()
    monkeypatch.setattr(factory, "SystemStateStorage", lambda: state, raising=True)

    built: list[object] = []

    def _build(self) -> object:
        obj = object()
        built.append(obj)
        return obj

    monkeypatch.setattr(factory.AppContainer, "build_rag_service", _build, raising=True)

    svc1 = await factory.get_rag_service()
    svc2 = await factory.get_rag_service()
    assert svc1 is svc2
    assert built == [svc1]

    # Simulate external invalidation (another process bumps shared version in DB).
    state.bump_version(factory.AppContainer.RAG_SERVICE_STATE_KEY)

    svc3 = await factory.get_rag_service()
    assert svc3 is not svc1
    assert built == [svc1, svc3]

    svc4 = await factory.get_rag_service()
    assert svc4 is svc3


def test_reset_rag_service_bumps_system_state_version(in_memory_sqlite, monkeypatch) -> None:
    state = SystemStateStorage(session_factory=in_memory_sqlite)
    factory.reset_app_context()
    monkeypatch.setattr(factory, "SystemStateStorage", lambda: state, raising=True)
    _ = factory.get_app_context()

    assert state.get_version(factory.AppContainer.RAG_SERVICE_STATE_KEY) == 0
    factory.reset_rag_service()
    assert state.get_version(factory.AppContainer.RAG_SERVICE_STATE_KEY) == 1

    factory.reset_rag_service()
    assert state.get_version(factory.AppContainer.RAG_SERVICE_STATE_KEY) == 2
