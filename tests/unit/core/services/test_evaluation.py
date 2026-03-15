# tests/unit/core/services/test_evaluation.py

from pathlib import Path

import pytest

from local_rag_backend.core.services.evaluation import (
    EvalDataset,
    compare_eval_results,
    eval_compare_result_to_json,
    load_eval_dataset,
    run_retrieval_eval,
)
from local_rag_backend.core.services.types import EvalResult


def test_load_eval_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_eval_dataset(tmp_path / "missing.jsonl")


def test_load_eval_dataset_rejects_unknown_type(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"nope"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown type"):
        load_eval_dataset(p)


def test_load_eval_dataset_rejects_non_integer_schema_version(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":"rag_eval_v1"}',
                '{"type":"doc","external_id":"doc:1","content":"alpha"}',
                '{"type":"query","query":"alpha","relevant_external_ids":["doc:1"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        load_eval_dataset(p)


def test_load_eval_dataset_rejects_duplicate_doc_external_ids(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc:1","content":"alpha"}',
                '{"type":"doc","external_id":" doc:1 ","content":"beta"}',
                '{"type":"query","query":"alpha","relevant_external_ids":["doc:1"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate doc external_id"):
        load_eval_dataset(p)


def test_load_eval_dataset_rejects_empty_relevant_external_ids(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc:1","content":"alpha"}',
                '{"type":"query","query":"alpha","relevant_external_ids":[]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not be empty"):
        load_eval_dataset(p)


def test_load_eval_dataset_rejects_relevant_ids_outside_corpus(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc:1","content":"alpha"}',
                '{"type":"query","query":"alpha","relevant_external_ids":["doc:missing"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside the corpus"):
        load_eval_dataset(p)


def test_load_eval_dataset_rejects_blank_ids_after_normalization(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"   ","content":"alpha"}',
                '{"type":"query","query":"alpha","relevant_external_ids":["doc:1"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="external_id must not be blank"):
        load_eval_dataset(p)


def test_run_retrieval_eval_allows_non_sparse_modes_when_callback_is_valid() -> None:
    ds = load_eval_dataset()
    result = run_retrieval_eval(
        dataset=ds,
        retrieve_external_ids=lambda _query, _top_k: (),
        retrieval_mode="dense",
        k=1,
    )

    assert result.retrieval_mode == "dense"
    assert result.queries == len(ds.queries)


def test_run_retrieval_eval_max_queries_zero_raises() -> None:
    ds: EvalDataset = load_eval_dataset()
    with pytest.raises(ValueError, match="No queries"):
        run_retrieval_eval(dataset=ds, retrieval_mode="sparse", max_queries=0)


def test_run_retrieval_eval_reports_standard_ir_metrics_for_perfect_run() -> None:
    ds = load_eval_dataset()

    def _retrieve_external_ids(query: str, top_k: int) -> tuple[str, ...]:
        if "France" in query:
            return ("doc:paris",)
        if "Spain" in query:
            return ("doc:madrid",)
        if "Germany" in query:
            return ("doc:berlin",)
        if "Portugal" in query:
            return ("doc:lisbon",)
        if "Italy" in query:
            return ("doc:rome",)
        return ()

    res = run_retrieval_eval(
        dataset=ds,
        retrieve_external_ids=_retrieve_external_ids,
        retrieval_mode="sparse",
        k=3,
    )

    assert res.ndcg_at_k == pytest.approx(1.0)
    assert res.map_at_k == pytest.approx(1.0)
    assert res.mrr_at_k == pytest.approx(1.0)
    assert 0.0 <= res.precision_at_k <= 1.0
    assert res.recall_at_k == pytest.approx(1.0)


def test_run_retrieval_eval_filters_unknown_ids_and_deduplicates_run() -> None:
    ds = EvalDataset(
        dataset_id="edge",
        schema_version=1,
        docs=(
            load_eval_dataset().docs[0],
            load_eval_dataset().docs[1],
        ),
        queries=(
            load_eval_dataset().queries[0],
            load_eval_dataset().queries[1],
        ),
    )

    def _retrieve_external_ids(query: str, top_k: int) -> tuple[str, ...]:
        if "France" in query:
            return ("doc:unknown", "doc:paris", "doc:paris")
        if "Spain" in query:
            return ("doc:paris", "doc:madrid")
        return ()

    res = run_retrieval_eval(
        dataset=ds,
        retrieve_external_ids=_retrieve_external_ids,
        retrieval_mode="sparse",
        k=2,
    )

    assert res.ndcg_at_k == pytest.approx(0.8154648767)
    assert res.map_at_k == pytest.approx(0.75)
    assert res.mrr_at_k == pytest.approx(0.75)
    assert res.precision_at_k == pytest.approx(0.5)
    assert res.recall_at_k == pytest.approx(1.0)


def _eval_result(
    *,
    retrieval_mode: str,
    ndcg: float,
    map_: float,
    mrr: float,
    precision: float,
    recall: float,
    reranker_enabled: bool = False,
) -> EvalResult:
    return EvalResult(
        dataset_id="x",
        retrieval_mode=retrieval_mode,
        reranker_enabled=reranker_enabled,
        k=3,
        queries=5,
        ndcg_at_k=ndcg,
        map_at_k=map_,
        mrr_at_k=mrr,
        precision_at_k=precision,
        recall_at_k=recall,
    )


def test_compare_eval_results_computes_deltas_and_passes_gate() -> None:
    baseline = _eval_result(
        retrieval_mode="sparse",
        ndcg=0.82,
        map_=0.74,
        mrr=0.78,
        precision=0.40,
        recall=0.86,
    )
    candidate = _eval_result(
        retrieval_mode="dual",
        ndcg=0.88,
        map_=0.80,
        mrr=0.84,
        precision=0.41,
        recall=0.89,
        reranker_enabled=True,
    )

    result = compare_eval_results(
        baseline=baseline,
        candidate=candidate,
        min_delta_ndcg=0.02,
        min_delta_map=0.02,
        min_delta_mrr=0.02,
        max_regression_precision=0.01,
        max_regression_recall=0.01,
    )

    assert result.gate.passed is True
    assert result.delta.ndcg_at_k == pytest.approx(0.06)
    assert result.delta.map_at_k == pytest.approx(0.06)
    assert result.delta.mrr_at_k == pytest.approx(0.06)
    assert result.delta.precision_at_k == pytest.approx(0.01)
    assert result.delta.recall_at_k == pytest.approx(0.03)
    payload = eval_compare_result_to_json(result)
    assert payload["gate"]["passed"] is True
    assert payload["delta"]["nDCG@3"] == pytest.approx(0.06)


def test_compare_eval_results_fails_when_candidate_does_not_improve_enough() -> None:
    baseline = _eval_result(
        retrieval_mode="sparse",
        ndcg=0.82,
        map_=0.74,
        mrr=0.78,
        precision=0.40,
        recall=0.86,
    )
    candidate = _eval_result(
        retrieval_mode="dense",
        ndcg=0.83,
        map_=0.74,
        mrr=0.79,
        precision=0.41,
        recall=0.86,
    )

    result = compare_eval_results(
        baseline=baseline,
        candidate=candidate,
        min_delta_ndcg=0.02,
        min_delta_map=0.02,
        min_delta_mrr=0.02,
    )

    assert result.gate.passed is False
    assert "nDCG@3" in result.gate.reasons[0]
    assert any("MAP@3" in reason for reason in result.gate.reasons)
    assert any("MRR@3" in reason for reason in result.gate.reasons)


def test_compare_eval_results_fails_when_precision_or_recall_regress_too_far() -> None:
    baseline = _eval_result(
        retrieval_mode="sparse",
        ndcg=0.82,
        map_=0.74,
        mrr=0.78,
        precision=0.40,
        recall=0.86,
    )
    candidate = _eval_result(
        retrieval_mode="hybrid",
        ndcg=0.90,
        map_=0.82,
        mrr=0.88,
        precision=0.37,
        recall=0.84,
    )

    result = compare_eval_results(
        baseline=baseline,
        candidate=candidate,
        min_delta_ndcg=0.02,
        min_delta_map=0.02,
        min_delta_mrr=0.02,
        max_regression_precision=0.01,
        max_regression_recall=0.01,
    )

    assert result.gate.passed is False
    assert any("P@3" in reason for reason in result.gate.reasons)
    assert any("Recall@3" in reason for reason in result.gate.reasons)
