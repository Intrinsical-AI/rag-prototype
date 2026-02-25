from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.app import factory

if TYPE_CHECKING:
    import pytest


def test_docs_mutation_ports_bind_factory_injected_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyRepo:
        class UpsertDoc:
            def __init__(self, **kwargs: object):
                self.kwargs = kwargs

    def _dummy_vec_factory(**kwargs: object) -> dict[str, object]:
        return kwargs

    def _dummy_rebuild(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr(factory, "SqlDocumentStorage", DummyRepo, raising=True)
    monkeypatch.setattr(factory, "VectorStorage", _dummy_vec_factory, raising=True)
    monkeypatch.setattr(factory, "rebuild_index_from_db", _dummy_rebuild, raising=True)

    container = factory._build_container()
    ports = container.docs_mutation_ports()

    assert isinstance(ports.doc_repo_factory(), DummyRepo)
    assert ports.build_upsert_doc is DummyRepo.UpsertDoc
    assert ports.vector_repo_factory is _dummy_vec_factory
    assert ports.rebuild_fn is _dummy_rebuild


def test_index_mutation_ports_bind_factory_injected_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyRepo:
        pass

    def _dummy_vec_factory(**kwargs: object) -> dict[str, object]:
        return kwargs

    def _dummy_purge(**kwargs: object) -> None:
        return None

    def _dummy_rebuild(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr(factory, "SqlDocumentStorage", DummyRepo, raising=True)
    monkeypatch.setattr(factory, "VectorStorage", _dummy_vec_factory, raising=True)
    monkeypatch.setattr(factory, "purge_index_artifacts", _dummy_purge, raising=True)
    monkeypatch.setattr(factory, "rebuild_index_from_db", _dummy_rebuild, raising=True)

    container = factory._build_container()
    ports = container.index_mutation_ports()

    assert isinstance(ports.doc_repo_factory(), DummyRepo)
    assert ports.vector_repo_factory is _dummy_vec_factory
    assert ports.purge_index_artifacts_fn is _dummy_purge
    assert ports.rebuild_fn is _dummy_rebuild
