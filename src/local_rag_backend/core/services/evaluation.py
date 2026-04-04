"""
Offline IR evaluation for retrieval quality.

Focus: reproducible retrieval metrics without requiring an LLM provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ir_measures
from ir_measures import AP, RR, P, R, nDCG

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


from local_rag_backend.core.services.types import (
    EvalCompareDelta,
    EvalCompareGate,
    EvalCompareResult,
    EvalDataset,
    EvalDoc,
    EvalQuery,
    EvalResult,
    EvalRetrievalMode,
)


def _default_eval_dataset_path() -> Path:
    # src/local_rag_backend/core/services/evaluation.py -> repo root
    return Path(__file__).resolve().parents[4] / "datasets" / "rag_eval_v1.jsonl"


def _parse_schema_version(raw_value: Any, *, lineno: int) -> int:
    try:
        return int(raw_value or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid dataset line {lineno}: schema_version must be an integer."
        ) from exc


def load_eval_dataset(path: str | Path | None = None) -> EvalDataset:
    content: str
    if path is None:
        p = _default_eval_dataset_path()
        if not p.is_file():
            raise FileNotFoundError(
                f"Default eval dataset not found: {p}. Pass --dataset or configure "
                "eval_dataset_path in config.yaml."
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
    known_doc_ids: set[str] = set()

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
            schema_version = _parse_schema_version(obj.get("schema_version"), lineno=lineno)
        elif t == "doc":
            external_id = str(obj["external_id"]).strip()
            if not external_id:
                raise ValueError(f"Invalid dataset line {lineno}: external_id must not be blank.")
            if external_id in known_doc_ids:
                raise ValueError(
                    f"Invalid dataset line {lineno}: duplicate doc external_id={external_id!r}."
                )
            known_doc_ids.add(external_id)
            docs.append(
                EvalDoc(
                    external_id=external_id,
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
            normalized_rel = tuple(str(external_id).strip() for external_id in rel)
            if not normalized_rel:
                raise ValueError(
                    f"Invalid dataset line {lineno}: relevant_external_ids must not be empty."
                )
            if any(not external_id for external_id in normalized_rel):
                raise ValueError(
                    f"Invalid dataset line {lineno}: relevant_external_ids must not contain blank IDs."
                )
            queries.append(
                EvalQuery(
                    query=str(obj["query"]),
                    relevant_external_ids=normalized_rel,
                )
            )
        else:
            raise ValueError(f"Invalid dataset line {lineno}: unknown type={t!r}")

    if schema_version != 1:
        raise ValueError(f"Unsupported schema_version={schema_version} (expected 1).")
    if not docs:
        raise ValueError("Dataset contains no docs.")
    if not queries:
        raise ValueError("Dataset contains no queries.")
    for query in queries:
        missing_ids = sorted(
            {
                external_id
                for external_id in query.relevant_external_ids
                if external_id not in known_doc_ids
            }
        )
        if missing_ids:
            raise ValueError(
                "Dataset query references relevant_external_ids outside the corpus: "
                + ", ".join(missing_ids[:10])
            )

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
    retrieval_mode: EvalRetrievalMode = "sparse",
    k: int = 3,
    reranker_enabled: bool = False,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
    run_out: Path | None = None,
) -> EvalResult:
    # Backward-compatible kwargs retained for callers migrating from the previous
    # infra-coupled implementation where reranker wiring happened in this layer.
    _ = (reranker_candidate_k, reranker_strategy)
    if k <= 0:
        raise ValueError("k must be positive")

    qs: list[EvalQuery] = list(dataset.queries)
    if max_queries is not None:
        qs = qs[: max(0, int(max_queries))]
    if not qs:
        raise ValueError("No queries to evaluate after max_queries.")
    if retrieve_external_ids is None:
        raise ValueError("retrieve_external_ids callback is required for offline IR evaluation.")
    known_doc_ids = {
        str(doc.external_id).strip() for doc in dataset.docs if str(doc.external_id).strip()
    }
    if not known_doc_ids:
        raise ValueError("Dataset contains no known corpus document IDs.")

    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for idx, q in enumerate(qs, start=1):
        query_id = f"q{idx:06d}"
        qrels[query_id] = {
            external_id: 1
            for external_id in q.relevant_external_ids
            if external_id in known_doc_ids
        }
        if not qrels[query_id]:
            raise ValueError(
                f"Query {query_id} has no relevant corpus IDs after dataset validation."
            )

        ranked_docs: dict[str, float] = {}
        seen_external_ids: set[str] = set()
        for external_id in retrieve_external_ids(q.query, k):
            normalized_external_id = str(external_id).strip()
            if (
                not normalized_external_id
                or normalized_external_id in seen_external_ids
                or normalized_external_id not in known_doc_ids
            ):
                continue
            seen_external_ids.add(normalized_external_id)
            rank = len(ranked_docs) + 1
            ranked_docs[normalized_external_id] = float(max(k - rank + 1, 1))
            if len(ranked_docs) >= int(k):
                break
        run[query_id] = ranked_docs

    if run_out is not None:
        run_out.parent.mkdir(parents=True, exist_ok=True)
        with run_out.open("w", encoding="utf-8") as _f:
            for idx, q in enumerate(qs, start=1):
                query_id = f"q{idx:06d}"
                ranked_list = sorted(run[query_id].items(), key=lambda x: -x[1])
                _f.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "query_text": q.query,
                            "retrieval_mode": retrieval_mode,
                            "k": k,
                            "ranked_docs": [
                                {"external_id": ext_id, "rank": rank, "score": score}
                                for rank, (ext_id, score) in enumerate(ranked_list, start=1)
                            ],
                        }
                    )
                    + "\n"
                )

    measures = (
        nDCG @ k,
        AP @ k,
        RR @ k,
        P @ k,
        R @ k,
    )
    aggregated = ir_measures.calc_aggregate(measures, qrels, run)
    n = len(qs)
    return EvalResult(
        dataset_id=dataset.dataset_id,
        retrieval_mode=retrieval_mode,
        reranker_enabled=bool(reranker_enabled),
        k=int(k),
        queries=n,
        ndcg_at_k=float(aggregated[nDCG @ k]),
        map_at_k=float(aggregated[AP @ k]),
        mrr_at_k=float(aggregated[RR @ k]),
        precision_at_k=float(aggregated[P @ k]),
        recall_at_k=float(aggregated[R @ k]),
    )


def format_eval_result(result: EvalResult) -> str:
    return (
        f"dataset={result.dataset_id} mode={result.retrieval_mode} reranker={result.reranker_enabled} "
        f"k={result.k} queries={result.queries} "
        f"nDCG@{result.k}={result.ndcg_at_k:.3f} "
        f"MAP@{result.k}={result.map_at_k:.3f} "
        f"MRR@{result.k}={result.mrr_at_k:.3f} "
        f"P@{result.k}={result.precision_at_k:.3f} "
        f"Recall@{result.k}={result.recall_at_k:.3f}"
    )


def _eval_metrics_to_json(result: EvalResult) -> dict[str, float]:
    return {
        f"nDCG@{result.k}": result.ndcg_at_k,
        f"MAP@{result.k}": result.map_at_k,
        f"MRR@{result.k}": result.mrr_at_k,
        f"P@{result.k}": result.precision_at_k,
        f"Recall@{result.k}": result.recall_at_k,
    }


def eval_result_to_json(result: EvalResult) -> dict[str, Any]:
    return {
        "dataset_id": result.dataset_id,
        "retrieval_mode": result.retrieval_mode,
        "reranker_enabled": result.reranker_enabled,
        "k": result.k,
        "queries": result.queries,
        "metrics": _eval_metrics_to_json(result),
    }


def compare_eval_results(
    *,
    baseline: EvalResult,
    candidate: EvalResult,
    min_delta_ndcg: float = 0.0,
    min_delta_map: float = 0.0,
    min_delta_mrr: float = 0.0,
    max_regression_precision: float = 0.0,
    max_regression_recall: float = 0.0,
) -> EvalCompareResult:
    if baseline.dataset_id != candidate.dataset_id:
        raise ValueError("Baseline and candidate dataset_id must match.")
    if baseline.k != candidate.k:
        raise ValueError("Baseline and candidate k must match.")

    delta = EvalCompareDelta(
        ndcg_at_k=float(candidate.ndcg_at_k - baseline.ndcg_at_k),
        map_at_k=float(candidate.map_at_k - baseline.map_at_k),
        mrr_at_k=float(candidate.mrr_at_k - baseline.mrr_at_k),
        precision_at_k=float(candidate.precision_at_k - baseline.precision_at_k),
        recall_at_k=float(candidate.recall_at_k - baseline.recall_at_k),
    )

    reasons: list[str] = []
    if delta.ndcg_at_k < float(min_delta_ndcg):
        reasons.append(
            f"nDCG@{baseline.k} delta={delta.ndcg_at_k:.3f} (min {float(min_delta_ndcg):.3f})"
        )
    if delta.map_at_k < float(min_delta_map):
        reasons.append(
            f"MAP@{baseline.k} delta={delta.map_at_k:.3f} (min {float(min_delta_map):.3f})"
        )
    if delta.mrr_at_k < float(min_delta_mrr):
        reasons.append(
            f"MRR@{baseline.k} delta={delta.mrr_at_k:.3f} (min {float(min_delta_mrr):.3f})"
        )
    if delta.precision_at_k < -abs(float(max_regression_precision)):
        reasons.append(
            f"P@{baseline.k} delta={delta.precision_at_k:.3f} "
            f"(max regression {float(max_regression_precision):.3f})"
        )
    if delta.recall_at_k < -abs(float(max_regression_recall)):
        reasons.append(
            f"Recall@{baseline.k} delta={delta.recall_at_k:.3f} "
            f"(max regression {float(max_regression_recall):.3f})"
        )

    return EvalCompareResult(
        dataset_id=baseline.dataset_id,
        k=int(baseline.k),
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        gate=EvalCompareGate(
            passed=not reasons,
            reasons=tuple(reasons),
            min_delta_ndcg=float(min_delta_ndcg),
            min_delta_map=float(min_delta_map),
            min_delta_mrr=float(min_delta_mrr),
            max_regression_precision=float(max_regression_precision),
            max_regression_recall=float(max_regression_recall),
        ),
    )


def format_eval_compare_result(result: EvalCompareResult) -> tuple[str, str, str, str]:
    k = int(result.k)
    baseline_line = "BASELINE  " + format_eval_result(result.baseline)
    candidate_line = "CANDIDATE " + format_eval_result(result.candidate)
    delta_line = (
        f"DELTA     nDCG@{k}={result.delta.ndcg_at_k:+.3f} "
        f"MAP@{k}={result.delta.map_at_k:+.3f} "
        f"MRR@{k}={result.delta.mrr_at_k:+.3f} "
        f"P@{k}={result.delta.precision_at_k:+.3f} "
        f"Recall@{k}={result.delta.recall_at_k:+.3f}"
    )
    gate_line = "PASS" if result.gate.passed else "FAIL: " + "; ".join(result.gate.reasons)
    return baseline_line, candidate_line, delta_line, gate_line


def eval_compare_result_to_json(result: EvalCompareResult) -> dict[str, Any]:
    return {
        "dataset_id": result.dataset_id,
        "k": result.k,
        "baseline": {
            "retrieval_mode": result.baseline.retrieval_mode,
            "reranker_enabled": result.baseline.reranker_enabled,
            "metrics": _eval_metrics_to_json(result.baseline),
        },
        "candidate": {
            "retrieval_mode": result.candidate.retrieval_mode,
            "reranker_enabled": result.candidate.reranker_enabled,
            "metrics": _eval_metrics_to_json(result.candidate),
        },
        "delta": {
            f"nDCG@{result.k}": result.delta.ndcg_at_k,
            f"MAP@{result.k}": result.delta.map_at_k,
            f"MRR@{result.k}": result.delta.mrr_at_k,
            f"P@{result.k}": result.delta.precision_at_k,
            f"Recall@{result.k}": result.delta.recall_at_k,
        },
        "gate": {
            "passed": result.gate.passed,
            "reasons": list(result.gate.reasons),
            "thresholds": {
                "min_delta_ndcg": result.gate.min_delta_ndcg,
                "min_delta_map": result.gate.min_delta_map,
                "min_delta_mrr": result.gate.min_delta_mrr,
                "max_regression_precision": result.gate.max_regression_precision,
                "max_regression_recall": result.gate.max_regression_recall,
            },
        },
    }
