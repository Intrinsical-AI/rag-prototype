"""
Offline evaluation for retrieval quality.

Focus: reproducible retrieval metrics without requiring an LLM provider.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


from local_rag_backend.core.services.types import EvalDataset, EvalDoc, EvalQuery, EvalResult

_EVAL_DATASET_ENV = "RAG_EVAL_DATASET_PATH"


def _default_eval_dataset_path() -> Path:
    env = str(os.getenv(_EVAL_DATASET_ENV, "")).strip()
    if env:
        return Path(env)
    # src/local_rag_backend/core/services/evaluation.py -> repo root
    return Path(__file__).resolve().parents[4] / "datasets" / "rag_eval_v1.jsonl"


def load_eval_dataset(path: str | Path | None = None) -> EvalDataset:
    content: str
    if path is None:
        p = _default_eval_dataset_path()
        if not p.is_file():
            raise FileNotFoundError(
                f"Default eval dataset not found: {p}. Set RAG_EVAL_DATASET_PATH or pass --dataset."
            )
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


def run_retrieval_eval(
    *,
    dataset: EvalDataset,
    retrieve_external_ids: Callable[[str, int], Sequence[str]] | None = None,
    retrieval_mode: str = "sparse",
    k: int = 3,
    reranker_enabled: bool = False,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
) -> EvalResult:
    # Backward-compatible kwargs retained for callers migrating from the previous
    # infra-coupled implementation where reranker wiring happened in this layer.
    _ = (reranker_candidate_k, reranker_strategy)
    if retrieval_mode != "sparse":
        raise ValueError(
            "This eval currently supports retrieval_mode=sparse only (dependency-free)."
        )
    if k <= 0:
        raise ValueError("k must be positive")

    qs: list[EvalQuery] = list(dataset.queries)
    if max_queries is not None:
        qs = qs[: max(0, int(max_queries))]
    if not qs:
        raise ValueError("No queries to evaluate after max_queries.")
    if retrieve_external_ids is None:
        raise ValueError("retrieve_external_ids callback is required for sparse evaluation.")

    hits = 0
    rr_sum = 0.0
    for q in qs:
        relevant = set(q.relevant_external_ids)
        retrieved_ext = [str(eid) for eid in retrieve_external_ids(q.query, k) if str(eid).strip()]

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
