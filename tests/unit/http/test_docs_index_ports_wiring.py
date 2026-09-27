from __future__ import annotations

from local_rag_backend.composition.container import AppContainer
from local_rag_backend.settings import Settings


def test_docs_mutation_ports_bind_explicit_container_factories(in_memory_sqlite) -> None:
    class DummyRepo:
        class UpsertDoc:
            def __init__(self, **kwargs: object):
                self.kwargs = kwargs

    def _dummy_vec_factory(**kwargs: object) -> dict[str, object]:
        return kwargs

    def _dummy_rebuild(**kwargs: object) -> int:
        return 0

    container = AppContainer(
        settings_obj=Settings(),
        session_factory=in_memory_sqlite,
        doc_repo_factory=DummyRepo,
        build_upsert_doc=DummyRepo.UpsertDoc,
        vector_repo_factory=_dummy_vec_factory,
        rebuild_fn=_dummy_rebuild,
    )
    ports = container.docs_mutation_ports()

    assert isinstance(ports.doc_repo_factory(), DummyRepo)
    assert ports.build_upsert_doc is DummyRepo.UpsertDoc
    assert ports.vector_repo_factory is _dummy_vec_factory
    container.close()


def test_index_mutation_ports_bind_explicit_container_factories(in_memory_sqlite) -> None:
    class DummyRepo:
        pass

    def _dummy_vec_factory(**kwargs: object) -> dict[str, object]:
        return kwargs

    def _dummy_purge(**kwargs: object) -> None:
        return None

    def _dummy_rebuild(**kwargs: object) -> int:
        return 0

    container = AppContainer(
        settings_obj=Settings(),
        session_factory=in_memory_sqlite,
        doc_repo_factory=DummyRepo,
        vector_repo_factory=_dummy_vec_factory,
        purge_index_artifacts_fn=_dummy_purge,
        rebuild_fn=_dummy_rebuild,
    )
    ports = container.index_mutation_ports()

    assert isinstance(ports.doc_repo_factory(), DummyRepo)
    assert ports.vector_repo_factory is _dummy_vec_factory
    assert ports.purge_index_artifacts_fn is _dummy_purge
    assert ports.rebuild_fn is _dummy_rebuild
    container.close()
