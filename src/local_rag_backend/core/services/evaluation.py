"""
Offline evaluation for retrieval quality.

Focus: reproducible retrieval metrics without requiring an LLM provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from local_rag_backend.core.services.corpus import get_corpus_and_ids
from local_rag_backend.core.services.reranking import RerankingRetriever
from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever

if TYPE_CHECKING:
    from local_rag_backend.core.ports import RetrieverPort


@dataclass(frozen=True)
class EvalDoc:
    external_id: str
    content: str
    source_id: str | None = None


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_external_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvalDataset:
    dataset_id: str
    schema_version: int
    docs: tuple[EvalDoc, ...]
    queries: tuple[EvalQuery, ...]


def load_eval_dataset(path: str | Path | None = None) -> EvalDataset:
    content: str
    if path is None:
        p = resources.files("local_rag_backend.datasets").joinpath("rag_eval_v1.jsonl")
        # Traversable (works for editable installs and packaged resources).
        content = p.read_text(encoding="utf-8")
    else:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Dataset not found: {p}")
        content = p.read_text(encoding="utf-8")

    dataset_id = "unknown"
    schema_version = 0
    docs: list[EvalDoc] = []
    queries: list[EvalQuery] = []

    for lineno, line in enumerate(content.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid dataset line {lineno}: expected JSON object.")
        t = obj.get("type")
        if t == "meta":
            dataset_id = str(obj.get("dataset_id") or dataset_id)
            schema_version = int(obj.get("schema_version") or 0)
        elif t == "doc":
            docs.append(
                EvalDoc(
                    external_id=str(obj["external_id"]),
                    content=str(obj["content"]),
                    source_id=(
                        str(obj.get("source_id")) if obj.get("source_id") is not None else None
                    ),
                )
            )
        elif t == "query":
            rel = obj.get("relevant_external_ids") or []
            if not isinstance(rel, list) or not all(isinstance(x, str) for x in rel):
                raise ValueError(
                    f"Invalid dataset line {lineno}: relevant_external_ids must be list[str]."
                )
            queries.append(EvalQuery(query=str(obj["query"]), relevant_external_ids=tuple(rel)))
        else:
            raise ValueError(f"Invalid dataset line {lineno}: unknown type={t!r}")

    if schema_version != 1:
        raise ValueError(f"Unsupported schema_version={schema_version} (expected 1).")
    if not docs:
        raise ValueError("Dataset contains no docs.")
    if not queries:
        raise ValueError("Dataset contains no queries.")

    return EvalDataset(
        dataset_id=str(dataset_id),
        schema_version=int(schema_version),
        docs=tuple(docs),
        queries=tuple(queries),
    )


@dataclass(frozen=True)
class EvalResult:
    dataset_id: str
    retrieval_mode: str
    reranker_enabled: bool
    k: int
    queries: int
    hit_rate: float
    mrr: float


def _build_ephemeral_doc_repo() -> SqlDocumentStorage:
    # StaticPool ensures all sessions share the same in-memory DB connection.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # Ensure models are registered with Base.metadata.
    from local_rag_backend.infrastructure.persistence.sqlalchemy import (  # noqa: F401
        models as _models,
    )

    db_base.Base.metadata.create_all(bind=engine)
    db_base.ensure_sqlite_documents_autoincrement(engine_to_use=engine)
    db_base.ensure_sqlite_documents_identity_columns(engine_to_use=engine)
    return SqlDocumentStorage(session_factory=session_local)


def _build_sparse_retriever(
    *, doc_repo: SqlDocumentStorage, reranker_enabled: bool, candidate_k: int, strategy: str
) -> RetrieverPort:
    corpus, doc_ids = get_corpus_and_ids(doc_repo)
    base: RetrieverPort = SparseBM25Retriever(documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo)
    if reranker_enabled:
        return RerankingRetriever(base, candidate_k=candidate_k, strategy=strategy)
    return base


def run_retrieval_eval(
    *,
    dataset: EvalDataset,
    retrieval_mode: str = "sparse",
    k: int = 3,
    reranker_enabled: bool = False,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
) -> EvalResult:
    if retrieval_mode != "sparse":
        raise ValueError(
            "This eval currently supports retrieval_mode=sparse only (dependency-free)."
        )
    if k <= 0:
        raise ValueError("k must be positive")

    doc_repo = _build_ephemeral_doc_repo()
    items = [
        SqlDocumentStorage.UpsertDoc(
            external_id=d.external_id,
            content=d.content,
            source_id=d.source_id,
            metadata={"dataset_id": dataset.dataset_id},
        )
        for d in dataset.docs
    ]
    results, _changed, _updated = doc_repo.upsert_documents_by_external_id(items)
    id_by_external_id = {r.external_id: int(r.id) for r in results}

    retriever = _build_sparse_retriever(
        doc_repo=doc_repo,
        reranker_enabled=reranker_enabled,
        candidate_k=reranker_candidate_k,
        strategy=reranker_strategy,
    )

    qs: list[EvalQuery] = list(dataset.queries)
    if max_queries is not None:
        qs = qs[: max(0, int(max_queries))]
    if not qs:
        raise ValueError("No queries to evaluate after max_queries.")

    hits = 0
    rr_sum = 0.0
    for q in qs:
        relevant = {eid for eid in q.relevant_external_ids if eid in id_by_external_id}
        docs, _scores = retriever.retrieve(q.query, k)
        retrieved_ext = [d.external_id for d in docs if getattr(d, "external_id", None)]

        hit = any(eid in relevant for eid in retrieved_ext)
        if hit:
            hits += 1

        rank = None
        for i, eid in enumerate(retrieved_ext, 1):
            if eid in relevant:
                rank = i
                break
        if rank is not None:
            rr_sum += 1.0 / float(rank)

    n = len(qs)
    return EvalResult(
        dataset_id=dataset.dataset_id,
        retrieval_mode=str(retrieval_mode),
        reranker_enabled=bool(reranker_enabled),
        k=int(k),
        queries=n,
        hit_rate=float(hits / n),
        mrr=float(rr_sum / n),
    )


def format_eval_result(result: EvalResult) -> str:
    return (
        f"dataset={result.dataset_id} mode={result.retrieval_mode} reranker={result.reranker_enabled} "
        f"k={result.k} queries={result.queries} hit_rate={result.hit_rate:.3f} mrr={result.mrr:.3f}"
    )


def eval_result_to_json(result: EvalResult) -> dict[str, Any]:
    return {
        "dataset_id": result.dataset_id,
        "retrieval_mode": result.retrieval_mode,
        "reranker_enabled": result.reranker_enabled,
        "k": result.k,
        "queries": result.queries,
        "hit_rate": result.hit_rate,
        "mrr": result.mrr,
    }
