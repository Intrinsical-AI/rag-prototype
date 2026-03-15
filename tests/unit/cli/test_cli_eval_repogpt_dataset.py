from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from local_rag_backend.cli import cli

DATASET_PATH = Path(__file__).resolve().parents[3] / "datasets" / "repogpt_rag_eval_v1.jsonl"


def test_rag_eval_repogpt_dataset_sparse_smoke() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "--dataset",
            str(DATASET_PATH),
            "--retrieval-mode",
            "sparse",
            "--k",
            "1",
            "--fail-below-ndcg",
            "0.0",
            "--fail-below-map",
            "0.0",
            "--fail-below-mrr",
            "0.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "dataset=repogpt_rag_eval_v1 mode=sparse" in result.output


def test_rag_eval_compare_repogpt_dataset_sparse_smoke(tmp_path: Path) -> None:
    json_out = tmp_path / "repogpt-compare.json"
    result = CliRunner().invoke(
        cli,
        [
            "eval-compare",
            "--dataset",
            str(DATASET_PATH),
            "--k",
            "1",
            "--candidate-mode",
            "sparse",
            "--min-delta-ndcg",
            "0.0",
            "--min-delta-map",
            "0.0",
            "--min-delta-mrr",
            "0.0",
            "--max-regression-precision",
            "0.0",
            "--max-regression-recall",
            "0.0",
            "--json-out",
            str(json_out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["dataset_id"] == "repogpt_rag_eval_v1"
    assert payload["gate"]["passed"] is True
