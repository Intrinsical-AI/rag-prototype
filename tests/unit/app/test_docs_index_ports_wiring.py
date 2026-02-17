from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.app import wiring

if TYPE_CHECKING:
    import pytest


def test_docs_mutation_ports_bind_current_router_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyRepo:
        class UpsertDoc:
            def __init__(self, **kwargs: object):
                self.kwargs = kwargs

    def _dummy_vec_factory(**kwargs: object) -> dict[str, object]:
        return kwargs

    def _dummy_precompute(**kwargs: object) -> dict[str, list[float]]:
        return {}

    def _dummy_sync(**kwargs: object) -> bool:
        return False

    def _dummy_rebuild(**kwargs: object) -> int:
        return 0

    def _dummy_delete_docs(**kwargs: object) -> tuple[int, int | None, bool]:
        return 0, None, False

    def _dummy_delete_external(**kwargs: object) -> tuple[int, int | None, list[str], int, bool]:
        return 0, None, [], 0, False

    monkeypatch.setattr(wiring, "SqlDocumentStorage", DummyRepo, raising=True)
    monkeypatch.setattr(wiring, "FaissVectorStorage", _dummy_vec_factory, raising=True)
    monkeypatch.setattr(wiring, "precompute_vectors_for_changed_items", _dummy_precompute, raising=True)
    monkeypatch.setattr(wiring, "sync_dense_after_upsert", _dummy_sync, raising=True)
    monkeypatch.setattr(wiring, "rebuild_index_from_db", _dummy_rebuild, raising=True)
    monkeypatch.setattr(wiring, "delete_documents_multi_store", _dummy_delete_docs, raising=True)
    monkeypatch.setattr(wiring, "delete_external_ids_multi_store", _dummy_delete_external, raising=True)

    ports = wiring.docs_mutation_ports()

    assert isinstance(ports.doc_repo_factory(), DummyRepo)
    assert ports.build_upsert_doc is DummyRepo.UpsertDoc
    assert ports.vector_repo_factory is _dummy_vec_factory
    assert ports.precompute_vectors_fn is _dummy_precompute
    assert ports.sync_dense_fn is _dummy_sync
    assert ports.rebuild_fn is _dummy_rebuild
    assert ports.delete_docs_fn is _dummy_delete_docs
    assert ports.delete_external_ids_fn is _dummy_delete_external


def test_index_mutation_ports_bind_current_router_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyRepo:
        pass

    def _dummy_vec_factory(**kwargs: object) -> dict[str, object]:
        return kwargs

    def _dummy_purge(**kwargs: object) -> None:
        return None

    def _dummy_rebuild(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr(wiring, "SqlDocumentStorage", DummyRepo, raising=True)
    monkeypatch.setattr(wiring, "FaissVectorStorage", _dummy_vec_factory, raising=True)
    monkeypatch.setattr(wiring, "purge_index_artifacts", _dummy_purge, raising=True)
    monkeypatch.setattr(wiring, "rebuild_index_from_db", _dummy_rebuild, raising=True)

    ports = wiring.index_mutation_ports()

    assert isinstance(ports.doc_repo_factory(), DummyRepo)
    assert ports.vector_repo_factory is _dummy_vec_factory
    assert ports.purge_index_artifacts_fn is _dummy_purge
    assert ports.rebuild_fn is _dummy_rebuild
