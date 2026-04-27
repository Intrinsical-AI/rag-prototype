"""IR metric calculation and retrieval-eval comparison."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import ir_measures
from ir_measures import AP, RR, P, R, nDCG

from local_rag_backend.core.services.evaluation_datasets import qrels_for_query
from local_rag_backend.core.services.evaluation_models import (
    EvalAnomaly,
    EvalCompareDelta,
    EvalCompareGate,
    EvalCompareQueryDelta,
    EvalCompareResult,
    EvalDataset,
    EvalQuery,
    EvalQueryMetrics,
    EvalQueryResult,
    EvalResult,
    EvalRetrievalMode,
    EvalRetrievedItem,
    EvalRunItem,
)
from local_rag_backend.core.services.evaluation_serialization import write_eval_run_jsonl


def _coerce_retrieved_item(raw_item: object, *, rank: int, k: int) -> EvalRetrievedItem:
    fallback_score = float(max(int(k) - int(rank) + 1, 1))
    if isinstance(raw_item, EvalRetrievedItem):
        return raw_item
    if isinstance(raw_item, str):
        return EvalRetrievedItem(external_id=raw_item, score=fallback_score)
    if isinstance(raw_item, tuple) and raw_item:
        external_id = str(raw_item[0])
        raw_score = raw_item[1] if len(raw_item) > 1 else fallback_score
        return EvalRetrievedItem(external_id=external_id, score=float(raw_score))
    if isinstance(raw_item, dict):
        external_id = str(raw_item.get("external_id") or raw_item.get("id") or "")
        raw_score = raw_item.get("score", fallback_score)
        return EvalRetrievedItem(external_id=external_id, score=float(raw_score))
    external_id = str(raw_item)
    return EvalRetrievedItem(external_id=external_id, score=fallback_score)


def _coerce_retrieved_items(
    raw_items: Sequence[object], *, k: int
) -> tuple[EvalRetrievedItem, ...]:
    return tuple(
        _coerce_retrieved_item(raw_item, rank=rank, k=k)
        for rank, raw_item in enumerate(raw_items, start=1)
    )


def _query_qrels_dict(query: EvalQuery) -> dict[str, int]:
    return {
        qrel.external_id: int(qrel.relevance)
        for qrel in qrels_for_query(query)
        if int(qrel.relevance) > 0
    }


def _metrics_from_aggregate(
    aggregated: Mapping[Any, float | int],
    *,
    k: int,
) -> EvalQueryMetrics:
    return EvalQueryMetrics(
        ndcg_at_k=float(aggregated[nDCG @ k]),
        map_at_k=float(aggregated[AP @ k]),
        mrr_at_k=float(aggregated[RR @ k]),
        precision_at_k=float(aggregated[P @ k]),
        recall_at_k=float(aggregated[R @ k]),
    )


def calculate_eval_metrics(
    *,
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int,
) -> EvalQueryMetrics:
    measures = (
        nDCG @ k,
        AP @ k,
        RR @ k,
        P @ k,
        R @ k,
    )
    aggregated = ir_measures.calc_aggregate(measures, qrels, run)
    return _metrics_from_aggregate(aggregated, k=k)


def run_retrieval_eval(
    *,
    dataset: EvalDataset,
    retrieve_external_ids: Callable[[str, int], Sequence[str]] | None = None,
    retrieve_ranked_items: Callable[[str, int], Sequence[object]] | None = None,
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
    if retrieve_external_ids is not None and retrieve_ranked_items is not None:
        raise ValueError("Pass only one of retrieve_external_ids or retrieve_ranked_items.")

    qs: list[EvalQuery] = list(dataset.queries)
    if max_queries is not None:
        qs = qs[: max(0, int(max_queries))]
    if not qs:
        raise ValueError("No queries to evaluate after max_queries.")
    if retrieve_external_ids is None and retrieve_ranked_items is None:
        raise ValueError("retrieve_ranked_items callback is required for offline IR evaluation.")
    known_doc_ids = {
        str(doc.external_id).strip() for doc in dataset.docs if str(doc.external_id).strip()
    }
    if not known_doc_ids:
        raise ValueError("Dataset contains no known corpus document IDs.")

    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    per_query_rows: dict[str, tuple[EvalRunItem, ...]] = {}
    per_query: list[EvalQueryResult] = []
    anomalies: list[EvalAnomaly] = []
    for idx, q in enumerate(qs, start=1):
        query_id = f"q{idx:06d}"
        query_qrels = _query_qrels_dict(q)
        qrels[query_id] = query_qrels
        if not qrels[query_id]:
            raise ValueError(
                f"Query {query_id} has no relevant corpus IDs after dataset validation."
            )

        ranked_docs: dict[str, float] = {}
        seen_external_ids: set[str] = set()
        query_run_items: list[EvalRunItem] = []
        if retrieve_ranked_items is not None:
            raw_ranked_items = retrieve_ranked_items(q.query, k)
        elif retrieve_external_ids is not None:
            raw_ranked_items = retrieve_external_ids(q.query, k)
        else:  # pragma: no cover - guarded above, keeps type checkers honest
            raw_ranked_items = ()
        for raw_item in _coerce_retrieved_items(raw_ranked_items, k=k):
            normalized_external_id = str(raw_item.external_id).strip()
            if not normalized_external_id:
                anomalies.append(
                    EvalAnomaly(
                        query_id=query_id,
                        kind="blank_external_id",
                        message="Retriever returned a blank external_id.",
                    )
                )
                continue
            if normalized_external_id in seen_external_ids:
                anomalies.append(
                    EvalAnomaly(
                        query_id=query_id,
                        kind="duplicate_external_id",
                        external_id=normalized_external_id,
                        message="Retriever returned a duplicate external_id.",
                    )
                )
                continue
            seen_external_ids.add(normalized_external_id)
            known = normalized_external_id in known_doc_ids
            anomaly_kind = None if known else "unknown_external_id"
            if not known:
                anomalies.append(
                    EvalAnomaly(
                        query_id=query_id,
                        kind="unknown_external_id",
                        external_id=normalized_external_id,
                        message="Retriever returned an external_id outside the eval corpus.",
                    )
                )
            score = float(
                raw_item.score if raw_item.score is not None else max(k - len(ranked_docs), 1)
            )
            ranked_docs[normalized_external_id] = score
            query_run_items.append(
                EvalRunItem(
                    external_id=normalized_external_id,
                    rank=len(query_run_items) + 1,
                    score=score,
                    known=known,
                    relevance=int(query_qrels.get(normalized_external_id, 0)),
                    anomaly=anomaly_kind,
                )
            )
            if len(ranked_docs) >= int(k):
                break
        run[query_id] = ranked_docs
        per_query_rows[query_id] = tuple(query_run_items)

    for idx, q in enumerate(qs, start=1):
        query_id = f"q{idx:06d}"
        query_metrics = calculate_eval_metrics(
            qrels={query_id: qrels[query_id]},
            run={query_id: run[query_id]},
            k=k,
        )
        per_query.append(
            EvalQueryResult(
                query_id=query_id,
                query_text=q.query,
                qrels=qrels_for_query(q),
                ranked_docs=per_query_rows[query_id],
                metrics=query_metrics,
            )
        )

    if run_out is not None:
        write_eval_run_jsonl(
            path=run_out,
            per_query=per_query,
            retrieval_mode=str(retrieval_mode),
            k=k,
        )

    aggregated_metrics = calculate_eval_metrics(qrels=qrels, run=run, k=k)
    n = len(qs)
    return EvalResult(
        dataset_id=dataset.dataset_id,
        retrieval_mode=retrieval_mode,
        reranker_enabled=bool(reranker_enabled),
        k=int(k),
        queries=n,
        ndcg_at_k=aggregated_metrics.ndcg_at_k,
        map_at_k=aggregated_metrics.map_at_k,
        mrr_at_k=aggregated_metrics.mrr_at_k,
        precision_at_k=aggregated_metrics.precision_at_k,
        recall_at_k=aggregated_metrics.recall_at_k,
        per_query=tuple(per_query),
        anomalies=tuple(anomalies),
    )


def compare_metrics_delta(
    *,
    baseline: EvalQueryMetrics,
    candidate: EvalQueryMetrics,
) -> EvalCompareDelta:
    return EvalCompareDelta(
        ndcg_at_k=float(candidate.ndcg_at_k - baseline.ndcg_at_k),
        map_at_k=float(candidate.map_at_k - baseline.map_at_k),
        mrr_at_k=float(candidate.mrr_at_k - baseline.mrr_at_k),
        precision_at_k=float(candidate.precision_at_k - baseline.precision_at_k),
        recall_at_k=float(candidate.recall_at_k - baseline.recall_at_k),
    )


def _result_metrics(result: EvalResult) -> EvalQueryMetrics:
    return EvalQueryMetrics(
        ndcg_at_k=result.ndcg_at_k,
        map_at_k=result.map_at_k,
        mrr_at_k=result.mrr_at_k,
        precision_at_k=result.precision_at_k,
        recall_at_k=result.recall_at_k,
    )


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

    delta = compare_metrics_delta(baseline=_result_metrics(baseline), candidate=_result_metrics(candidate))

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

    baseline_by_query = {query.query_id: query for query in baseline.per_query}
    candidate_by_query = {query.query_id: query for query in candidate.per_query}
    per_query = tuple(
        EvalCompareQueryDelta(
            query_id=query_id,
            baseline=baseline_by_query[query_id].metrics,
            candidate=candidate_by_query[query_id].metrics,
            delta=compare_metrics_delta(
                baseline=baseline_by_query[query_id].metrics,
                candidate=candidate_by_query[query_id].metrics,
            ),
        )
        for query_id in sorted(baseline_by_query.keys() & candidate_by_query.keys())
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
        per_query=per_query,
    )
