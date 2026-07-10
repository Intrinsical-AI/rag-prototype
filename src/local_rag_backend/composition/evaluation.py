"""Composition adapters for isolated offline evaluation workspaces."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from local_rag_backend.composition.embeddings import (
    DEFAULT_DENSE_BACKEND_MESSAGE,
    _build_default_openai_embedder,
    _build_default_st_embedder,
    build_dense_embedder_from_settings,
)
from local_rag_backend.core.domain.retrieval import RetrievalRequest, RetrievalResult
from local_rag_backend.core.ports import (
    EmbedderPort,
    EvalDatasetDocInput,
    EvalRetrieverFactoryPort,
    EvalRetrieverPort,
    EvalStoragePort,
    RetrieverPort,
)
from local_rag_backend.core.services.evaluation_models import EvalRetrievalConfig
from local_rag_backend.core.services.reranking import RerankingRetriever
from local_rag_backend.infrastructure.persistence.sql import (
    SqlDocumentStorage,
    base as db_base,
)
from local_rag_backend.infrastructure.persistence.vector.manifest import (
    expected_manifest_config_from_settings,
)
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.infrastructure.retrieval.dense_vector import DenseVectorRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import (
    HybridRetriever,
    combine_hybrid_results,
)
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.infrastructure.search_backends import LocalSplitSearchRetriever

if TYPE_CHECKING:
    from local_rag_backend.settings import Settings

# Keep eval rebuild batches bounded so large offline datasets do not materialize every
# document embedding in memory at once.
EVAL_DENSE_REBUILD_BATCH_SIZE = 64


class _SqlEvalStoragePort(EvalStoragePort):
    """Evaluation storage adapter for offline/e2e workflows.

    The port isolates evaluation from production persistence by creating a dedicated
    local SQLite-backed workspace per dataset signature.
    """

    def __init__(self, *, settings_obj: Settings) -> None:
        self._base_settings = settings_obj
        self._workspace_root = Path(settings_obj.data_dir).resolve() / "_eval_workspaces"
        self._active_dataset_signature: str | None = None
        self._active_dataset_id: str | None = None
        self._active_external_ids: tuple[str, ...] = ()
        self._eval_settings = self._settings_for_eval_root(self._workspace_root / "_pending")
        self._doc_repo: SqlDocumentStorage | None = None

    def _get_doc_repo(self) -> SqlDocumentStorage:
        """Return the active dataset repository, initializing it when first used."""
        if self._doc_repo is None:
            self._doc_repo = self._new_doc_repo()
        return self._doc_repo

    def _new_doc_repo(self) -> SqlDocumentStorage:
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)

        # Side-effect import: registers SQLAlchemy model metadata with db_base.
        from local_rag_backend.infrastructure.persistence.sql import models as _models  # noqa: F401

        db_base.ensure_sqlite_schema_compatible(engine_to_use=engine)
        return SqlDocumentStorage(session_factory=session_local)

    def _settings_for_eval_root(self, eval_root: Path) -> Settings:
        return self._base_settings.model_copy(
            update={
                "persistence_backend": "local_split",
                "search_backend": "local_split",
                "data_dir": eval_root,
                "index_path": str(eval_root / "eval.index"),
                "id_map_path": str(eval_root / "eval_id_map.json"),
                "sqlite_url": f"sqlite:///{eval_root / 'eval.db'}",
            }
        )

    def _dataset_signature(
        self,
        *,
        dataset_id: str,
        docs: tuple[EvalDatasetDocInput, ...],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(str(dataset_id).encode("utf-8"))
        for doc in docs:
            digest.update(b"\0")
            digest.update(str(doc.external_id).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(doc.source_id or "").encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                hashlib.sha256(str(doc.content).encode("utf-8")).hexdigest().encode("ascii")
            )
        return digest.hexdigest()

    def _workspace_slug(self, dataset_id: str) -> str:
        collapsed = "".join(ch if ch.isalnum() else "-" for ch in str(dataset_id).strip().lower())
        normalized = "-".join(part for part in collapsed.split("-") if part)
        return normalized[:48] or "dataset"

    def _workspace_root_for_dataset(self, *, dataset_id: str, signature: str) -> Path:
        slug = self._workspace_slug(dataset_id)
        return self._workspace_root / f"{slug}-{signature[:16]}"

    def get_dataset_signature(self) -> str | None:
        return self._active_dataset_signature

    def upsert_dataset_docs(
        self,
        *,
        dataset_id: str,
        docs: tuple[EvalDatasetDocInput, ...],
    ) -> tuple[str, ...]:
        signature = self._dataset_signature(dataset_id=dataset_id, docs=docs)
        external_ids = tuple(str(doc.external_id) for doc in docs)
        if (
            self._active_dataset_id == str(dataset_id)
            and self._active_dataset_signature == signature
            and self._active_external_ids == external_ids
        ):
            return external_ids

        eval_root = self._workspace_root_for_dataset(
            dataset_id=str(dataset_id), signature=signature
        )
        eval_root.mkdir(parents=True, exist_ok=True)
        self._eval_settings = self._settings_for_eval_root(eval_root)
        self._doc_repo = self._new_doc_repo()
        items = [
            SqlDocumentStorage.UpsertDoc(
                external_id=d.external_id,
                content=d.content,
                source_id=d.source_id,
                metadata={"dataset_id": dataset_id, **(d.metadata or {})},
            )
            for d in docs
        ]
        results, _changed, _updated = self._doc_repo.upsert_documents_by_external_id(items)
        self._active_dataset_id = str(dataset_id)
        self._active_dataset_signature = signature
        self._active_external_ids = external_ids
        return tuple(str(r.external_id) for r in results if r.external_id is not None)

    def list_documents(self) -> tuple[Any, ...]:
        return tuple(self._get_doc_repo().get_all_documents())

    def get_retriever_storage(self) -> Any:
        return self._get_doc_repo()

    def get_eval_settings(self) -> Any:
        return self._eval_settings


class _QueryCachingEmbedder(EmbedderPort):
    """Cache exact single-text query embeddings within a prepared eval workspace."""

    def __init__(self, base: EmbedderPort) -> None:
        self._base = base
        self.dim = base.dim
        self._cache: dict[str, Any] = {}

    def embed(self, texts: Sequence[str]) -> Sequence[Any]:
        if len(texts) != 1:
            return self._base.embed(texts)
        text = str(texts[0])
        cached = self._cache.get(text)
        if cached is not None:
            return [cached]
        vector = self._base.embed(texts)[0]
        self._cache[text] = vector
        return [vector]


class _PreparedEvalWorkspace:
    """Prepared, immutable snapshot used by eval retriever assembly."""

    def __init__(
        self,
        *,
        storage: EvalStoragePort,
        openai_embedder_factory: Callable[[], EmbedderPort],
        st_embedder_factory: Callable[[str], EmbedderPort],
        dense_retriever_factory: Callable[..., RetrieverPort],
        hybrid_retriever_factory: Callable[..., RetrieverPort],
        vector_repo_factory: Callable[..., Any],
        reranker_factory: Callable[..., Any],
    ) -> None:
        self._storage = storage
        self._docs = self._snapshot_documents(storage=storage)
        self._doc_repo = storage.get_retriever_storage()
        self._eval_settings = storage.get_eval_settings()
        self._dataset_signature = (
            getattr(storage, "get_dataset_signature", lambda: None)()
            if callable(getattr(storage, "get_dataset_signature", None))
            else None
        )
        self._openai_embedder_factory = openai_embedder_factory
        self._st_embedder_factory = st_embedder_factory
        self._dense_retriever_factory = dense_retriever_factory
        self._hybrid_retriever_factory = hybrid_retriever_factory
        self._vector_repo_factory = vector_repo_factory
        self._reranker_factory = reranker_factory
        self._query_embedder: EmbedderPort | None = None
        self._vector_repo: Any | None = None
        self._vector_repo_ready = False
        self._sparse_retriever: RetrieverPort | None = None
        self._dense_retriever: RetrieverPort | None = None

    def _snapshot_documents(self, *, storage: EvalStoragePort) -> tuple[Any, ...]:
        """Take a stable document snapshot for the active dataset revision."""
        return tuple(storage.list_documents())

    def _workspace_manifest_path(self) -> Path:
        return Path(self._eval_settings.data_dir) / "eval_workspace_manifest.json"

    def _doc_ids_signature(self) -> str:
        digest = hashlib.sha256()
        for doc in self._docs:
            digest.update(str(doc.id).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _expected_workspace_manifest(self) -> dict[str, object]:
        return {
            "dataset_signature": self._dataset_signature,
            "doc_ids_signature": self._doc_ids_signature(),
            "doc_count": len(self._docs),
            "index_path": str(self._eval_settings.index_path),
            "id_map_path": str(self._eval_settings.id_map_path),
            "vector_backend": str(getattr(self._eval_settings, "vector_backend", "auto")),
            "vector_manifest_config": expected_manifest_config_from_settings(self._eval_settings),
        }

    def _workspace_manifest_matches(self) -> bool:
        path = self._workspace_manifest_path()
        if not path.exists():
            return False
        if not Path(self._eval_settings.index_path).exists():
            return False
        if not Path(self._eval_settings.id_map_path).exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return False
        return isinstance(payload, dict) and payload == self._expected_workspace_manifest()

    def _write_workspace_manifest(self) -> None:
        path = self._workspace_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._expected_workspace_manifest(), indent=2) + "\n",
            encoding="utf-8",
        )

    def _dense_embedder(self) -> EmbedderPort:
        if self._query_embedder is not None:
            return self._query_embedder
        base = build_dense_embedder_from_settings(
            settings_obj=self._eval_settings,
            openai_embedder_factory=self._openai_embedder_factory,
            st_embedder_factory=self._st_embedder_factory,
            missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE,
        )
        self._query_embedder = _QueryCachingEmbedder(base)
        return self._query_embedder

    def _dense_vector_repo(self) -> Any:
        if self._vector_repo is not None:
            return self._vector_repo
        embedder = self._dense_embedder()
        self._vector_repo = self._vector_repo_factory(
            index_path=self._eval_settings.index_path,
            id_map_path=self._eval_settings.id_map_path,
            dim=embedder.dim,
            backend=getattr(self._eval_settings, "vector_backend", "auto"),
            settings_obj=self._eval_settings,
        )
        return self._vector_repo

    def _ensure_vector_repo_ready(self) -> Any:
        vector_repo = self._dense_vector_repo()
        if self._vector_repo_ready:
            return vector_repo
        if self._workspace_manifest_matches():
            self._vector_repo_ready = True
            return vector_repo
        if self._docs:
            embedder = self._dense_embedder()
            rebuild_batches: list[tuple[list[Any], Sequence[Any]]] = []
            for start in range(0, len(self._docs), EVAL_DENSE_REBUILD_BATCH_SIZE):
                chunk = self._docs[start : start + EVAL_DENSE_REBUILD_BATCH_SIZE]
                chunk_ids = [doc.id for doc in chunk]
                chunk_embeddings = embedder.embed([doc.content for doc in chunk])
                rebuild_batches.append((chunk_ids, chunk_embeddings))
            rebuild_from_batches = getattr(vector_repo, "rebuild_from_batches", None)
            if callable(rebuild_from_batches):
                rebuild_from_batches(rebuild_batches)
            else:  # pragma: no cover
                doc_ids = [doc.id for doc in self._docs]
                embeddings = [vector for _ids, vectors in rebuild_batches for vector in vectors]
                vector_repo.rebuild(doc_ids, embeddings)
            self._write_workspace_manifest()
        self._vector_repo_ready = True
        return vector_repo

    def _cached_sparse_retriever(self) -> RetrieverPort:
        if self._sparse_retriever is None:
            self._sparse_retriever = SparseBM25Retriever(
                documents=[doc.content for doc in self._docs],
                doc_ids=[doc.id for doc in self._docs],
                doc_repo=self._doc_repo,
                preloaded_docs=self._docs,
            )
        return self._sparse_retriever

    def _build_eval_mode_retriever(self, *, config: EvalRetrievalConfig) -> RetrieverPort:
        mode = str(config.retrieval_mode)
        if mode in {"sparse", "dense", "dual"}:
            return LocalSplitSearchRetriever(
                doc_repo=self._doc_repo,
                embedder=(self._dense_embedder() if mode in {"dense", "dual"} else None),
                vector_repo=(self._ensure_vector_repo_ready() if mode == "dense" else None),
                preloaded_docs=self._docs,
                cached_sparse_retriever=cast(
                    "SparseBM25Retriever", self._cached_sparse_retriever()
                ),
            )

        dense_retriever = self._cached_dense_retriever()
        sparse_retriever = self._cached_sparse_retriever()
        return self._hybrid_retriever_factory(
            dense=dense_retriever,
            sparse=sparse_retriever,
            alpha=self._select_hybrid_alpha(config=config),
        )

    def _select_hybrid_alpha(self, *, config: EvalRetrievalConfig) -> float:
        """Resolve hybrid alpha with config override precedence."""
        return (
            config.hybrid_alpha
            if config.hybrid_alpha is not None
            else self._eval_settings.hybrid_retrieval_alpha
        )

    def _cached_dense_retriever(self) -> RetrieverPort:
        if self._dense_retriever is None:
            self._dense_retriever = self._dense_retriever_factory(
                embedder=self._dense_embedder(),
                vector_repo=self._ensure_vector_repo_ready(),
                doc_repo=self._doc_repo,
            )
        return self._dense_retriever

    def build_retriever(
        self,
        *,
        config: EvalRetrievalConfig,
        reranker_candidate_k: int,
        reranker_strategy: str,
    ) -> EvalRetrieverPort:
        mode = str(config.retrieval_mode)
        if mode not in {"sparse", "dense", "dual", "hybrid"}:
            raise ValueError(f"Unsupported retrieval_mode: {mode}")

        retriever = self._build_eval_mode_retriever(config=config)

        if config.reranker_enabled:
            retriever = self._reranker_factory(
                retriever,
                candidate_k=int(reranker_candidate_k),
                strategy=str(reranker_strategy),
            )
        return cast("EvalRetrieverPort", retriever)

    def retrieve_hybrid_alpha_group(
        self,
        *,
        query: str,
        top_k: int,
        alphas: Sequence[float],
    ) -> dict[float, RetrievalResult]:
        if not alphas:
            return {}
        dense_result = self._cached_dense_retriever().retrieve(
            RetrievalRequest(query=query, top_k=top_k, mode="dense")
        )
        sparse_result = self._cached_sparse_retriever().retrieve(
            RetrievalRequest(query=query, top_k=top_k, mode="sparse")
        )
        return {
            float(alpha): combine_hybrid_results(
                dense_result=dense_result,
                sparse_result=sparse_result,
                alpha=float(alpha),
                top_k=top_k,
            )
            for alpha in alphas
        }


@dataclass(frozen=True)
class _DefaultEvalRetrieverFactoryPort(EvalRetrieverFactoryPort):
    openai_embedder_factory: Callable[[], EmbedderPort]
    st_embedder_factory: Callable[[str], EmbedderPort]
    sparse_retriever_factory: Callable[..., RetrieverPort]
    dense_retriever_factory: Callable[..., RetrieverPort]
    hybrid_retriever_factory: Callable[..., RetrieverPort]
    vector_repo_factory: Callable[..., Any]
    reranker_factory: Callable[..., Any]

    def build_retriever(
        self,
        *,
        storage: EvalStoragePort,
        config: EvalRetrievalConfig,
        reranker_candidate_k: int,
        reranker_strategy: str,
    ) -> EvalRetrieverPort:
        workspace = self.prepare_workspace(storage=storage)
        return workspace.build_retriever(
            config=config,
            reranker_candidate_k=reranker_candidate_k,
            reranker_strategy=reranker_strategy,
        )

    def prepare_workspace(self, *, storage: EvalStoragePort) -> _PreparedEvalWorkspace:
        return _PreparedEvalWorkspace(
            storage=storage,
            openai_embedder_factory=self.openai_embedder_factory,
            st_embedder_factory=self.st_embedder_factory,
            dense_retriever_factory=self.dense_retriever_factory,
            hybrid_retriever_factory=self.hybrid_retriever_factory,
            vector_repo_factory=self.vector_repo_factory,
            reranker_factory=self.reranker_factory,
        )


def build_eval_storage_port(
    *,
    settings_obj: Settings,
) -> EvalStoragePort:
    return _SqlEvalStoragePort(settings_obj=settings_obj)


def build_eval_retriever_factory_port(
    *,
    openai_embedder_factory: Callable[[], EmbedderPort] | None = None,
    st_embedder_factory: Callable[[str], EmbedderPort] | None = None,
    sparse_retriever_factory: Callable[..., RetrieverPort] = SparseBM25Retriever,
    dense_retriever_factory: Callable[..., RetrieverPort] = DenseVectorRetriever,
    hybrid_retriever_factory: Callable[..., RetrieverPort] = HybridRetriever,
    vector_repo_factory: Callable[..., Any] = VectorStorage,
    reranker_factory: Callable[..., Any] = RerankingRetriever,
) -> EvalRetrieverFactoryPort:
    return _DefaultEvalRetrieverFactoryPort(
        openai_embedder_factory=openai_embedder_factory or _build_default_openai_embedder,
        st_embedder_factory=st_embedder_factory or _build_default_st_embedder,
        sparse_retriever_factory=sparse_retriever_factory,
        dense_retriever_factory=dense_retriever_factory,
        hybrid_retriever_factory=hybrid_retriever_factory,
        vector_repo_factory=vector_repo_factory,
        reranker_factory=reranker_factory,
    )
