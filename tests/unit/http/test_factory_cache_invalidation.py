from __future__ import annotations

import pytest

from local_rag_backend.composition import factory
from local_rag_backend.composition.container import AppContainer
from local_rag_backend.composition.context import AppContext
from local_rag_backend.infrastructure.persistence.sql import SystemStateStorage
from local_rag_backend.settings import Settings


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
    _ = factory.get_app_context()

    assert state.get_version(factory.AppContainer.RAG_SERVICE_STATE_KEY) == 0
    factory.reset_rag_service()
    assert state.get_version(factory.AppContainer.RAG_SERVICE_STATE_KEY) == 1

    factory.reset_rag_service()
    assert state.get_version(factory.AppContainer.RAG_SERVICE_STATE_KEY) == 2


def test_reset_rag_service_keeps_owned_memory_database(monkeypatch, tmp_path, reset_app_context):
    settings = Settings(sqlite_url="sqlite:///:memory:", data_dir=tmp_path)
    created = []

    def build_context():
        container = AppContainer.from_settings(settings)
        container.initialize()
        created.append(container)
        return AppContext(settings_obj=settings, container=container)

    monkeypatch.setattr(factory, "_build_app_context", build_context)
    first = factory.get_app_context()
    first.container.doc_repo_factory().store_documents(["keep this document"])
    factory.reset_rag_service()
    second = factory.get_app_context()
    assert second is first
    assert created == [first.container]
    assert [doc.content for doc in second.container.doc_repo_factory().get_all_documents()] == [
        "keep this document"
    ]
    assert second.container.read_rag_service_version() == 1
