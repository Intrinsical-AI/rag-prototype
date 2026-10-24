"""Compose snapshot-only reads without constructing the mutable application container."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from local_rag_backend.composition.adapters import (
    build_generator_from_settings,
    build_retriever_from_settings,
)
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalFilter, document_matches_filters
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.ports import EmbedderPort
from local_rag_backend.core.services.rag_runtime import RagService
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator
from local_rag_backend.infrastructure.persistence.read_snapshot import SnapshotError, load_snapshot
from local_rag_backend.infrastructure.persistence.vector.read_snapshot import SnapshotVectorStorage
from local_rag_backend.settings import Settings


@dataclass(frozen=True)
class SnapshotDocuments:
    documents: tuple[Document, ...]

    def get_all_documents(self) -> tuple[Document, ...]:
        return self.documents

    def get(self, ids: Sequence[DocId]) -> list[Document]:
        by_id = {doc.id: doc for doc in self.documents}
        return [by_id[item] for item in ids if item in by_id]

    def query_documents(
        self, *, limit: int, offset: int, filters: tuple[RetrievalFilter, ...] = ()
    ) -> list[Document]:
        return [doc for doc in self.documents if document_matches_filters(doc, filters)][
            offset : offset + limit
        ]

    def store_documents(self, contents: Sequence[str]) -> Sequence[DocId]:
        del contents
        raise SnapshotError("Frozen documents are read-only")

    def delete_documents(self, ids: Sequence[DocId]) -> None:
        del ids
        raise SnapshotError("Frozen documents are read-only")


def _embedder(settings: Settings) -> EmbedderPort:
    if settings.openai_api_key:
        return OpenAIEmbedder(settings_obj=settings, max_retries=0)
    if not settings.synthetic_embeddings:
        path = Path(settings.st_embedding_model).expanduser()
        if not path.is_absolute() or not path.is_dir() or not (path / "modules.json").is_file():
            raise SnapshotError("Ephemeral embeddings require an installed local model directory")
    try:
        return SentenceTransformerEmbedder(
            model_name=settings.st_embedding_model, settings_obj=settings, local_only=True
        )
    except RuntimeError as exc:
        raise SnapshotError(
            "Installed local embeddings are unavailable; verify offline startup and the dense-st extra"
        ) from exc


@contextmanager
def build_ephemeral_service(settings: Settings) -> Iterator[tuple[RagService, str]]:
    snapshot = load_snapshot(settings)
    vectors = SnapshotVectorStorage(snapshot) if snapshot.vector_bytes is not None else None
    repository = SnapshotDocuments(snapshot.documents)

    with ExitStack() as resources:

        def dense_embedder() -> EmbedderPort:
            embedder = _embedder(settings)
            if isinstance(embedder, OpenAIEmbedder):
                resources.callback(embedder.client.close)
            if vectors is not None and embedder.dim != vectors.dim:
                raise SnapshotError("Snapshot query embedding dimension drift")
            if snapshot.native_manifest is not None and any(
                snapshot.native_manifest.get(key) != value
                for key, value in embedder.identity.manifest_fields().items()
            ):
                raise SnapshotError("Snapshot query embedding identity drift")
            return embedder

        retriever = build_retriever_from_settings(
            settings_obj=settings,
            retrieval_mode=str(settings.retrieval_mode),
            doc_repo=repository,
            dense_embedder_factory=dense_embedder,
            preloaded_docs=snapshot.documents,
            vector_repo_factory=lambda **_kwargs: vectors,
        )
        generator = build_generator_from_settings(
            settings_obj=settings,
            openai_generator_factory=partial(OpenAIGenerator, settings_obj=settings, max_retries=0),
        )
        if isinstance(generator, OpenAIGenerator):
            resources.callback(generator.client.close)
        yield (
            RagService(retriever=retriever, generator=generator, history_storage=None),
            snapshot.manifest.snapshot_id,
        )


def snapshot_status(settings: Settings) -> dict[str, object]:
    """No live SQL query, schema initialization, journal inspection or model loading."""
    live_path = settings.get_database_path()
    health: dict[str, object] = {
        "snapshot_ready": False,
        "snapshot_id": None,
        "documents_count": None,
    }
    if settings.ephemeral_snapshot_dir is None:
        return {
            "runtime": {
                "retrieval_mode": str(settings.retrieval_mode),
                "live_store": "present_uninspected" if live_path.is_file() else "missing",
                "snapshot_freshness": "frozen_generation_only",
            },
            "health": {**health, "snapshot_state": "not_configured"},
        }
    try:
        snapshot = load_snapshot(settings)
        if snapshot.vector_bytes is not None:
            SnapshotVectorStorage(snapshot)
        health.update(
            snapshot_ready=True,
            snapshot_id=snapshot.manifest.snapshot_id,
            documents_count=len(snapshot.documents),
        )
    except (SnapshotError, OSError, ValueError):
        health["snapshot_error"] = (
            "No verified snapshot available; configure or prepare a new generation"
        )
    return {
        "runtime": {
            "retrieval_mode": str(settings.retrieval_mode),
            "live_store": "present_uninspected" if live_path.is_file() else "missing",
            "snapshot_freshness": "frozen_generation_only",
        },
        "health": health,
    }
