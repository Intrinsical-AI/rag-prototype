from __future__ import annotations

from typing import TYPE_CHECKING, Any

from local_rag_backend.composition.container import AppContainer
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    import pytest


class DummyEmbedder:
    dim = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def test_docs_mutation_ports_use_injected_dependencies() -> None:
    def _doc_repo_factory() -> object:
        return object()

    class _UpsertDoc:
        def __init__(self, **kwargs: object):
            self.kwargs = kwargs

    def _vec_factory(**kwargs: object) -> dict[str, object]:
        return kwargs

    def _rebuild(**kwargs: object) -> int:
        return 7

    container = AppContainer(
        settings_obj=settings,
        doc_repo_factory=_doc_repo_factory,
        build_upsert_doc=_UpsertDoc,
        vector_repo_factory=_vec_factory,
        rebuild_fn=_rebuild,
    )

    ports = container.docs_mutation_ports(build_embedder=lambda: DummyEmbedder())

    assert ports.doc_repo_factory is _doc_repo_factory
    assert ports.build_upsert_doc is _UpsertDoc
    assert ports.vector_repo_factory is _vec_factory
    assert ports.rebuild_fn is _rebuild


def test_docs_mutation_ports_defaults_are_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySqlRepo:
        class UpsertDoc:
            pass

    from local_rag_backend.composition import container as container_module

    monkeypatch.setattr(container_module, "SqlDocumentStorage", DummySqlRepo, raising=True)

    container = AppContainer(settings_obj=settings)
    ports = container.docs_mutation_ports(build_embedder=lambda: DummyEmbedder())

    assert isinstance(ports.doc_repo_factory(), DummySqlRepo)
    assert ports.build_upsert_doc is DummySqlRepo.UpsertDoc
    assert ports.vector_repo_factory is container.vector_repo_factory


def test_index_mutation_ports_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySqlRepo:
        class UpsertDoc:
            pass

    def _purge(**kwargs: object) -> None:
        return None

    def _rebuild(**kwargs: object) -> int:
        return 3

    from local_rag_backend.composition import container as container_module

    monkeypatch.setattr(container_module, "SqlDocumentStorage", DummySqlRepo, raising=True)

    default_container = AppContainer(settings_obj=settings)
    default_ports = default_container.index_mutation_ports(build_embedder=lambda: DummyEmbedder())
    assert isinstance(default_ports.doc_repo_factory(), DummySqlRepo)
    assert default_ports.vector_repo_factory is default_container.vector_repo_factory

    def _repo_factory() -> Any:
        return "repo"

    def _vec_factory(**kwargs: object) -> str:
        return "vec"

    custom_container = AppContainer(
        settings_obj=settings,
        doc_repo_factory=_repo_factory,
        vector_repo_factory=_vec_factory,
        purge_index_artifacts_fn=_purge,
        rebuild_fn=_rebuild,
    )
    custom_ports = custom_container.index_mutation_ports(build_embedder=lambda: DummyEmbedder())

    assert custom_ports.doc_repo_factory is _repo_factory
    assert custom_ports.vector_repo_factory is _vec_factory
    assert custom_ports.purge_index_artifacts_fn is _purge
    assert custom_ports.rebuild_fn is _rebuild


def test_from_settings_preserves_injectable_overrides() -> None:
    class _Repo:
        class UpsertDoc:
            pass

    class _CustomUpsert:
        pass

    def _doc_repo_factory() -> _Repo:
        return _Repo()

    container = AppContainer.from_settings(
        settings,
        doc_repo_factory=_doc_repo_factory,
        build_upsert_doc=_CustomUpsert,
    )
    ports = container.docs_mutation_ports(build_embedder=lambda: DummyEmbedder())

    assert isinstance(ports.doc_repo_factory(), _Repo)
    assert ports.build_upsert_doc is _CustomUpsert


def test_elasticsearch_defaults_bind_elastic_runtime() -> None:
    es_settings = settings.model_copy(
        update={
            "persistence_backend": "elasticsearch",
            "retrieval_mode": "dense",
            "es_base_url": "http://localhost:9200",
        }
    )

    class _DummySystemState:
        def get_version(self, key: str) -> int:
            _ = key
            return 0

        def bump_version(self, key: str) -> int:
            _ = key
            return 1

    container = AppContainer(
        settings_obj=es_settings,
        system_state_factory=_DummySystemState,
    )
    ports = container.docs_mutation_ports(build_embedder=lambda: DummyEmbedder())

    assert ports.build_upsert_doc.__qualname__.endswith("ElasticDocsRepository.UpsertDoc")
    assert ports.mutation_uow_factory is None
    assert container.doc_repo_factory.__name__ == "<lambda>"
    assert container.history_repo_factory.__name__ == "<lambda>"
    assert container.vector_repo_factory.__name__ == "<lambda>"
