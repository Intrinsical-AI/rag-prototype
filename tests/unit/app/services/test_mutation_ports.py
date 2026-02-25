from __future__ import annotations

from typing import TYPE_CHECKING, Any

from local_rag_backend.app.services.mutation_ports import (
    build_build_index_ports,
    build_docs_mutation_ports,
    build_index_mutation_ports,
)

if TYPE_CHECKING:
    import pytest


class DummyEmbedder:
    dim = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def test_build_docs_mutation_ports_uses_injected_dependencies() -> None:
    def _build_embedder() -> DummyEmbedder:
        return DummyEmbedder()

    def _doc_repo_factory() -> object:
        return object()

    class _UpsertDoc:
        def __init__(self, **kwargs: object):
            self.kwargs = kwargs

    def _vec_factory(**kwargs: object) -> dict[str, object]:
        return kwargs

    def _precompute(**kwargs: object) -> dict[str, list[float]]:
        return {}

    def _sync(**kwargs: object) -> bool:
        return True

    def _rebuild(**kwargs: object) -> int:
        return 7

    def _delete_docs(**kwargs: object) -> tuple[int, int | None, bool]:
        return 1, 1, False

    def _delete_external(**kwargs: object) -> tuple[int, int | None, list[str], int, bool]:
        return 1, 1, [], 1, False

    ports = build_docs_mutation_ports(
        build_embedder=_build_embedder,
        doc_repo_factory=_doc_repo_factory,
        build_upsert_doc=_UpsertDoc,
        vector_repo_factory=_vec_factory,
        precompute_vectors_fn=_precompute,
        sync_dense_fn=_sync,
        rebuild_fn=_rebuild,
        delete_docs_fn=_delete_docs,
        delete_external_ids_fn=_delete_external,
    )

    assert ports.doc_repo_factory is _doc_repo_factory
    assert ports.build_upsert_doc is _UpsertDoc
    assert ports.vector_repo_factory is _vec_factory
    assert ports.precompute_vectors_fn is _precompute
    assert ports.sync_dense_fn is _sync
    assert ports.rebuild_fn is _rebuild
    assert ports.delete_docs_fn is _delete_docs
    assert ports.delete_external_ids_fn is _delete_external


def test_build_docs_mutation_ports_defaults_are_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySqlRepo:
        class UpsertDoc:
            pass

    class DummyVec:
        pass

    from local_rag_backend.infrastructure.persistence.sql import alchemy_engine as sql_module
    from local_rag_backend.infrastructure.persistence.vector import storage as vector_module

    monkeypatch.setattr(sql_module, "SqlDocumentStorage", DummySqlRepo, raising=True)
    monkeypatch.setattr(vector_module, "VectorStorage", DummyVec, raising=True)

    ports = build_docs_mutation_ports(build_embedder=lambda: DummyEmbedder())

    assert isinstance(ports.doc_repo_factory(), DummySqlRepo)
    assert ports.build_upsert_doc is DummySqlRepo.UpsertDoc
    assert ports.vector_repo_factory is DummyVec


def test_build_index_mutation_ports_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySqlRepo:
        pass

    class DummyVec:
        pass

    def _purge(**kwargs: object) -> None:
        return None

    def _rebuild(**kwargs: object) -> int:
        return 3

    from local_rag_backend.infrastructure.persistence.sql import alchemy_engine as sql_module
    from local_rag_backend.infrastructure.persistence.vector import storage as vector_module

    monkeypatch.setattr(sql_module, "SqlDocumentStorage", DummySqlRepo, raising=True)
    monkeypatch.setattr(vector_module, "VectorStorage", DummyVec, raising=True)

    default_ports = build_index_mutation_ports(build_embedder=lambda: DummyEmbedder())
    assert isinstance(default_ports.doc_repo_factory(), DummySqlRepo)
    assert default_ports.vector_repo_factory is DummyVec

    def _repo_factory() -> Any:
        return "repo"

    def _vec_factory(**kwargs: object) -> str:
        return "vec"

    custom_ports = build_index_mutation_ports(
        build_embedder=lambda: DummyEmbedder(),
        doc_repo_factory=_repo_factory,
        vector_repo_factory=_vec_factory,
        purge_index_artifacts_fn=_purge,
        rebuild_fn=_rebuild,
    )
    assert custom_ports.doc_repo_factory is _repo_factory
    assert custom_ports.vector_repo_factory is _vec_factory
    assert custom_ports.purge_index_artifacts_fn is _purge
    assert custom_ports.rebuild_fn is _rebuild


def test_build_build_index_ports_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    def _default_runner(**kwargs: object) -> int:
        return 5

    monkeypatch.setattr(
        "local_rag_backend.scripts.sample_data_ingestion.run_sample_data_ingestion",
        _default_runner,
        raising=True,
    )

    default_ports = build_build_index_ports()
    assert default_ports.run_sample_data_ingestion_fn is _default_runner

    def _custom_runner(**kwargs: object) -> int:
        return 7

    custom_ports = build_build_index_ports(run_sample_data_ingestion_fn=_custom_runner)
    assert custom_ports.run_sample_data_ingestion_fn is _custom_runner
