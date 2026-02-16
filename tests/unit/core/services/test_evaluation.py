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


def test_run_retrieval_eval_rejects_non_sparse_mode() -> None:
    ds = load_eval_dataset()
    with pytest.raises(ValueError, match="sparse only"):
        run_retrieval_eval(dataset=ds, retrieval_mode="dense")


def test_run_retrieval_eval_max_queries_zero_raises() -> None:
    ds: EvalDataset = load_eval_dataset()
    with pytest.raises(ValueError, match="No queries"):
        run_retrieval_eval(dataset=ds, retrieval_mode="sparse", max_queries=0)
