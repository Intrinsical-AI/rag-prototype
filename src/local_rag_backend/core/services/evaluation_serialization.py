"""Evaluation result formatting and JSON serialization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_rag_backend.core.services.evaluation_models import (
    EvalAnomaly,
    EvalCompareResult,
    EvalQrel,
    EvalQueryMetrics,
    EvalQueryResult,
    EvalResult,
    EvalRunItem,
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


def eval_metrics_to_json(result: EvalResult) -> dict[str, float]:
    return {
        f"nDCG@{result.k}": result.ndcg_at_k,
        f"MAP@{result.k}": result.map_at_k,
        f"MRR@{result.k}": result.mrr_at_k,
        f"P@{result.k}": result.precision_at_k,
        f"Recall@{result.k}": result.recall_at_k,
    }


def eval_query_metrics_to_json(metrics: EvalQueryMetrics, *, k: int) -> dict[str, float]:
    return {
        f"nDCG@{k}": metrics.ndcg_at_k,
        f"MAP@{k}": metrics.map_at_k,
        f"MRR@{k}": metrics.mrr_at_k,
        f"P@{k}": metrics.precision_at_k,
        f"Recall@{k}": metrics.recall_at_k,
    }


def eval_anomaly_to_json(anomaly: EvalAnomaly) -> dict[str, Any]:
    return {
        "query_id": anomaly.query_id,
        "kind": anomaly.kind,
        "external_id": anomaly.external_id,
        "message": anomaly.message,
    }


def eval_run_item_to_json(item: EvalRunItem) -> dict[str, Any]:
    return {
        "external_id": item.external_id,
        "rank": item.rank,
        "score": item.score,
        "known": item.known,
        "relevance": item.relevance,
        "anomaly": item.anomaly,
    }


def eval_qrel_to_json(qrel: EvalQrel) -> dict[str, Any]:
    return {"external_id": qrel.external_id, "relevance": qrel.relevance}


def eval_query_result_to_json(query_result: EvalQueryResult) -> dict[str, Any]:
    return {
        "query_id": query_result.query_id,
        "query_text": query_result.query_text,
        "qrels": [eval_qrel_to_json(qrel) for qrel in query_result.qrels],
        "ranked_docs": [eval_run_item_to_json(item) for item in query_result.ranked_docs],
        "metrics": {
            "nDCG": query_result.metrics.ndcg_at_k,
            "MAP": query_result.metrics.map_at_k,
            "MRR": query_result.metrics.mrr_at_k,
            "P": query_result.metrics.precision_at_k,
            "Recall": query_result.metrics.recall_at_k,
        },
    }


def eval_result_to_json(result: EvalResult, *, include_details: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_id": result.dataset_id,
        "retrieval_mode": result.retrieval_mode,
        "reranker_enabled": result.reranker_enabled,
        "k": result.k,
        "queries": result.queries,
        "metrics": eval_metrics_to_json(result),
    }
    if include_details:
        payload["anomalies_count"] = len(result.anomalies)
        payload["per_query"] = [
            {
                **eval_query_result_to_json(query_result),
                "metrics": eval_query_metrics_to_json(query_result.metrics, k=result.k),
            }
            for query_result in result.per_query
        ]
        payload["anomalies"] = [eval_anomaly_to_json(anomaly) for anomaly in result.anomalies]
    return payload


def eval_result_summary_to_json(result: EvalResult) -> dict[str, object]:
    return {
        "dataset_id": result.dataset_id,
        "retrieval_mode": result.retrieval_mode,
        "reranker_enabled": result.reranker_enabled,
        "k": result.k,
        "queries": result.queries,
        "ndcg_at_k": result.ndcg_at_k,
        "map_at_k": result.map_at_k,
        "mrr_at_k": result.mrr_at_k,
        "precision_at_k": result.precision_at_k,
        "recall_at_k": result.recall_at_k,
        "anomalies_count": len(result.anomalies),
    }


def eval_result_report_to_json(
    result: EvalResult,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "eval_report",
        "schema_version": 1,
        "config": dict(config or {}),
        **eval_result_to_json(result, include_details=True),
    }


def eval_anomalies_to_jsonl(result: EvalResult) -> str:
    return "\n".join(json.dumps(eval_anomaly_to_json(anomaly)) for anomaly in result.anomalies)


def write_eval_run_jsonl(
    *, path: Path, per_query: Sequence[EvalQueryResult], retrieval_mode: str, k: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for query_result in per_query:
            out.write(
                json.dumps(
                    {
                        "query_id": query_result.query_id,
                        "query_text": query_result.query_text,
                        "retrieval_mode": retrieval_mode,
                        "k": k,
                        "qrels": [
                            eval_qrel_to_json(qrel)
                            for qrel in query_result.qrels
                            if qrel.relevance > 0
                        ],
                        "ranked_docs": [
                            eval_run_item_to_json(item) for item in query_result.ranked_docs
                        ],
                    }
                )
                + "\n"
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


def eval_compare_result_to_json(
    result: EvalCompareResult,
    *,
    include_details: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_id": result.dataset_id,
        "k": result.k,
        "baseline": {
            "retrieval_mode": result.baseline.retrieval_mode,
            "reranker_enabled": result.baseline.reranker_enabled,
            "metrics": eval_metrics_to_json(result.baseline),
        },
        "candidate": {
            "retrieval_mode": result.candidate.retrieval_mode,
            "reranker_enabled": result.candidate.reranker_enabled,
            "metrics": eval_metrics_to_json(result.candidate),
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
    if include_details:
        payload["baseline"]["anomalies_count"] = len(result.baseline.anomalies)
        payload["candidate"]["anomalies_count"] = len(result.candidate.anomalies)
        payload["per_query"] = [
            {
                "query_id": item.query_id,
                "baseline": eval_query_metrics_to_json(item.baseline, k=result.k),
                "candidate": eval_query_metrics_to_json(item.candidate, k=result.k),
                "delta": {
                    f"nDCG@{result.k}": item.delta.ndcg_at_k,
                    f"MAP@{result.k}": item.delta.map_at_k,
                    f"MRR@{result.k}": item.delta.mrr_at_k,
                    f"P@{result.k}": item.delta.precision_at_k,
                    f"Recall@{result.k}": item.delta.recall_at_k,
                },
            }
            for item in result.per_query
        ]
    return payload


def eval_compare_report_to_json(result: EvalCompareResult) -> dict[str, Any]:
    return {
        "type": "eval_compare_report",
        "schema_version": 1,
        **eval_compare_result_to_json(result, include_details=True),
    }
