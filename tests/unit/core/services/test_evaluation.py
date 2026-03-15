# tests/unit/core/services/test_evaluation.py

from pathlib import Path

import pytest

from local_rag_backend.core.services.evaluation import (
    EvalDataset,
    load_eval_dataset,
    run_retrieval_eval,
)


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
