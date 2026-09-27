from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import nullcontext
from threading import Event

import pytest

from local_rag_backend.composition.container import AppContainer
from local_rag_backend.settings import Settings


@pytest.fixture()
def containers(tmp_path):
    result = []
    for name, dimension in (("left", 3), ("right", 5)):
        cfg = Settings(
            sqlite_url=f"sqlite:///{tmp_path / (name + '.db')}",
            data_dir=tmp_path / name,
            synthetic_embeddings=True,
            synthetic_embedding_dim=dimension,
            st_embedding_model=name,
            openai_api_key=None,
            disable_embedding_cache=True,
        )
        container = AppContainer.from_settings(cfg)
        container.initialize()
        result.append(container)
    try:
        yield tuple(result)
    finally:
        for container in result:
            container.close()


def test_containers_isolate_documents_history_state_and_embedding_config(containers):
    left, right = containers
    for container, name, dimension in ((left, "left", 3), (right, "right", 5)):
        docs = container.doc_repo_factory()
        docs.upsert_documents_by_external_id(
            [
                container.build_upsert_doc(
                    external_id="shared", content=name, metadata={"scope": name}
                )
            ]
        )
        container.history_repo_factory().save(name, "answer", [])
        embedder = container.build_dense_embedder()
        assert embedder.identity.model == name
        assert embedder.identity.dimension == dimension
        assert len(embedder.embed(["same input"])[0]) == dimension

    for container, name in ((left, "left"), (right, "right")):
        assert [doc.content for doc in container.doc_repo_factory().get_all_documents()] == [name]
        history = container.build_history_read_port().list_history_entries(limit=10, offset=0)
        assert [entry.question for entry in history] == [name]

    left.reset_rag_service()
    assert left.read_rag_service_version() == 1
    assert right.read_rag_service_version() == 0


def test_container_transaction_scope_does_not_capture_another_database(containers):
    left, right = containers
    with pytest.raises(RuntimeError, match="rollback left"), left.mutation_uow_factory():
        left.doc_repo_factory().store_documents(["rolled back"])
        right.doc_repo_factory().store_documents(["committed independently"])
        raise RuntimeError("rollback left")
    assert left.doc_repo_factory().get_all_documents() == []
    assert [doc.content for doc in right.doc_repo_factory().get_all_documents()] == [
        "committed independently"
    ]


@pytest.mark.parametrize("rollback", [False, True])
def test_memory_database_serializes_concurrent_sessions(tmp_path, rollback):
    container = AppContainer.from_settings(
        Settings(sqlite_url="sqlite:///:memory:", data_dir=tmp_path)
    )
    container.initialize()
    repo = container.doc_repo_factory()
    reader_started = Event()

    def read_documents():
        reader_started.set()
        return [doc.content for doc in repo.get_all_documents()]

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            expected_error = (
                pytest.raises(RuntimeError, match="rollback") if rollback else nullcontext()
            )
            with expected_error, container.mutation_uow_factory():
                repo.store_documents(["pending document"])
                reader = executor.submit(read_documents)
                assert reader_started.wait(timeout=2)
                # A second session must wait for the transaction's connection:
                # it must neither read uncommitted data nor roll back that write.
                with pytest.raises(TimeoutError):
                    reader.result(timeout=0.05)
                if rollback:
                    raise RuntimeError("rollback")
            expected = [] if rollback else ["pending document"]
            assert reader.result(timeout=2) == expected
            assert read_documents() == expected
    finally:
        container.close()
