"""Compose live, generator-free reads without bootstrapping mutable application state."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from local_rag_backend.composition.adapters import (
    build_retriever_with_default_embedder_from_settings,
)
from local_rag_backend.core.domain.retrieval import RetrievalRequest
from local_rag_backend.core.errors import MutationRecoveryRequiredError, WriteLockTimeoutError
from local_rag_backend.core.use_cases.search import execute_search
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal
from local_rag_backend.infrastructure.persistence.sql.document_storage import SqlDocumentStorage
from local_rag_backend.infrastructure.persistence.vector.index import VectorIndex
from local_rag_backend.infrastructure.persistence.vector.manifest import (
    manifest_path_for,
    read_manifest,
)
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage

if TYPE_CHECKING:
    from local_rag_backend.core.domain.retrieval import RetrievalResult
    from local_rag_backend.settings import Settings


@contextmanager
def _existing_read_lock(path: Path, *, timeout: float, required: bool) -> Iterator[None]:
    """Share a writer's existing lock; a query never creates a coordination file."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        if required:
            raise MutationRecoveryRequiredError(
                "Vector reads require an existing writer lock"
            ) from None
        yield
        return
    try:
        try:
            import fcntl
        except ImportError:
            fcntl = None  # type: ignore[assignment]
        if fcntl is None:
            import msvcrt

            def acquire() -> None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBRLCK, 1)

            def release() -> None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:

            def acquire() -> None:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)

            def release() -> None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)

        deadline = time.monotonic() + timeout
        while True:
            try:
                acquire()
                break
            except (BlockingIOError, PermissionError):
                if time.monotonic() >= deadline:
                    raise WriteLockTimeoutError(
                        "Live retrieval is waiting for a document mutation"
                    ) from None
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            release()
    finally:
        os.close(descriptor)


class _ExistingVectorStorage(VectorStorage):
    """Use current manifest identity and read paths; never run the writer constructor."""

    def __init__(self, *, settings_obj: Settings, **kwargs: Any) -> None:
        self._settings = settings_obj
        self._embedding_identity = kwargs.get("embedding_identity")
        manifest = read_manifest(manifest_path_for(settings_obj.index_path))
        if manifest is None:
            raise ValueError("Vector manifest missing; rebuild the private index first")
        backend, dimension = manifest.get("index_backend"), manifest.get("dimension")
        if backend not in {"numpy", "faiss"} or type(dimension) is not int:
            raise ValueError("Vector manifest backend or dimension is invalid")
        self.vector_index = VectorIndex.open_existing(
            settings_obj.index_path, settings_obj.id_map_path, dim=dimension, backend=backend
        )
        self._ensure_manifest(overwrite=False, create=False)


def search_live(*, settings: Settings, request: RetrievalRequest) -> RetrievalResult:
    """Read one current private corpus; generation/history/schema initialization are absent.

    Sparse reads need no embedding provider. Dense modes preserve the configured query
    embedder and its egress; embedding caches are disabled for this query composition.
    """
    database = settings.get_database_path().absolute()
    if not database.is_file():
        raise FileNotFoundError("Private document store missing; import or mutate documents first")
    coordination = settings.get_coordination_dir()
    with _existing_read_lock(
        coordination / ".rag_multi_store_write.lock",
        timeout=settings.write_lock_timeout_s,
        required=request.mode != "sparse",
    ):
        journal = FileMutationJournal(coordination / ".mutation_journal")
        if journal.count_incomplete():
            raise MutationRecoveryRequiredError(
                "Private document mutation needs recovery before search"
            )
        url = URL.create("sqlite", database=f"file:{database}", query={"mode": "ro", "uri": "true"})
        engine = create_engine(url, poolclass=NullPool)
        try:
            repository = SqlDocumentStorage(sessionmaker(bind=engine))
            documents = tuple(repository.get_all_documents())
            read_settings = settings.model_copy(update={"disable_embedding_cache": True})
            retriever = build_retriever_with_default_embedder_from_settings(
                settings_obj=read_settings,
                retrieval_mode=request.mode,
                doc_repo=repository,
                preloaded_docs=documents,
                vector_repo_factory=_ExistingVectorStorage,
            )
            return execute_search(retriever=retriever, request=request)
        finally:
            engine.dispose()
