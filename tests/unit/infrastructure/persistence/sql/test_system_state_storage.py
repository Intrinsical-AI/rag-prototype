from __future__ import annotations

import pytest
from sqlalchemy import text

from local_rag_backend.composition.container import AppContainer
from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
from local_rag_backend.core.use_cases.docs_mutation import (
    MutationCoordinator,
    MutationIntent,
    MutationUpsertInput,
)
from local_rag_backend.infrastructure.persistence.sql import SystemStateStorage
from local_rag_backend.infrastructure.persistence.sql.base import session_uow
from local_rag_backend.settings import Settings


def test_system_state_get_version_defaults_to_zero(in_memory_sqlite) -> None:
    st = SystemStateStorage(session_factory=in_memory_sqlite)
    assert st.get_version("rag_service") == 0


def test_system_state_bump_version_is_monotonic(in_memory_sqlite) -> None:
    st = SystemStateStorage(session_factory=in_memory_sqlite)

    assert st.bump_version("rag_service") == 1
    assert st.bump_version("rag_service") == 2
    assert st.get_version("rag_service") == 2


def test_system_state_bump_rolls_back_with_document_mutation(in_memory_sqlite) -> None:
    st = SystemStateStorage(session_factory=in_memory_sqlite)
    assert st.get_version("rag_service") == 0

    with (
        pytest.raises(RuntimeError, match="abort mutation"),
        session_uow(session_factory=in_memory_sqlite),
    ):
        assert st.bump_version("rag_service") == 1
        raise RuntimeError("abort mutation")

    assert st.get_version("rag_service") == 0


def test_saga_bumps_shared_cache_version_inside_sql_uow(in_memory_sqlite, tmp_path) -> None:
    container = AppContainer(
        settings_obj=Settings(retrieval_mode="sparse", data_dir=tmp_path),
        session_factory=in_memory_sqlite,
    )
    container.initialize()
    coordinator = MutationCoordinator(
        settings_obj=container.settings_obj,
        ports=container.docs_mutation_ports(),
    )

    assert container.read_rag_service_version() == 0
    coordinator.execute(
        MutationIntent(
            op_id="versioned-write",
            upserts=(MutationUpsertInput(external_id="doc:1", content="raw text"),),
        )
    )

    assert container.read_rag_service_version() == 1
    assert [doc.content for doc in container.doc_repo_factory().get_all_documents()] == ["raw text"]
    container.close()


def test_dense_write_republishes_version_after_vector_delta(in_memory_sqlite, tmp_path) -> None:
    class Embedder:
        dim = 2
        identity = EmbeddingIdentity("sentence_transformers", "test", 2, synthetic=True)

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class VectorRepo:
        during_delta: object | None = None

        def apply_delta_atomic(self, *, delete_ids, upserts):
            self.during_delta = container.get_rag_service()

    vector = VectorRepo()
    container = AppContainer(
        settings_obj=Settings(retrieval_mode="dense", data_dir=tmp_path),
        session_factory=in_memory_sqlite,
        vector_repo_factory=lambda **kwargs: vector,
    )
    container.initialize()
    container.build_rag_service = lambda: object()  # type: ignore[method-assign]
    before = container.get_rag_service()
    coordinator = MutationCoordinator(
        settings_obj=container.settings_obj,
        ports=container.docs_mutation_ports(build_embedder=lambda: Embedder()),
    )

    coordinator.execute(
        MutationIntent(
            op_id="dense-versioned-write",
            upserts=(MutationUpsertInput(external_id="doc:1", content="raw text"),),
        )
    )

    assert vector.during_delta is not before
    assert container.read_rag_service_version() == 2
    assert container.get_rag_service() is not vector.during_delta
    container.close()


def test_system_state_storage_creates_table_on_demand(in_memory_sqlite) -> None:
    st = SystemStateStorage(session_factory=in_memory_sqlite)
    st.bump_version("rag_service")

    with in_memory_sqlite() as session:
        row = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='system_state'")
        ).first()
        assert row is not None


def test_system_state_storage_rejects_blank_keys(in_memory_sqlite) -> None:
    st = SystemStateStorage(session_factory=in_memory_sqlite)

    with pytest.raises(ValueError, match="must not be blank"):
        st.get_version("   ")

    with pytest.raises(ValueError, match="must not be blank"):
        st.bump_version("")
