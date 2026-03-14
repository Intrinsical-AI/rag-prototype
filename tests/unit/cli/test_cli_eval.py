# tests/unit/test_cli_eval.py

import json
from pathlib import Path

from click.testing import CliRunner

from local_rag_backend.cli import cli


def test_rag_eval_fails_below_threshold(tmp_path: Path) -> None:
    # Valid dataset where the retriever prefers a non-relevant doc => thresholds fail.
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1,"created_at":"2026-02-16"}',
                '{"type":"doc","external_id":"doc:1","source_id":"eval","content":"zzz zzz zzz"}',
                '{"type":"doc","external_id":"doc:2","source_id":"eval","content":"alpha alpha alpha"}',
                '{"type":"query","query":"alpha","relevant_external_ids":["doc:1"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    r = CliRunner().invoke(
        cli,
        [
            "eval",
            "--dataset",
            str(ds),
            "--retrieval-mode",
            "sparse",
            "--k",
            "1",
            "--fail-below-ndcg",
            "0.5",
        ],
    )
    assert r.exit_code == 1, r.output
    assert "Eval regression" in r.output
    assert "dataset=x mode=sparse" in r.output
    assert r.output.count("\n") == 1


def test_rag_eval_reports_invalid_jsonl_cleanly(tmp_path: Path) -> None:
    ds = tmp_path / "bad.jsonl"
    ds.write_text('{"not":"valid"\n', encoding="utf-8")

    r = CliRunner().invoke(cli, ["eval", "--dataset", str(ds)])
    assert r.exit_code == 1
    assert "[ERROR] Error evaluating dataset:" in r.output


def test_rag_eval_reports_empty_dataset_cleanly(tmp_path: Path) -> None:
    ds = tmp_path / "empty.jsonl"
    ds.write_text("", encoding="utf-8")

    r = CliRunner().invoke(cli, ["eval", "--dataset", str(ds)])
    assert r.exit_code == 1
    assert "[ERROR] Error evaluating dataset:" in r.output


def test_rag_eval_json_out_uses_metrics_only_shape(tmp_path: Path) -> None:
    ds = tmp_path / "ds.jsonl"
    json_out = tmp_path / "result.json"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1,"created_at":"2026-02-16"}',
                '{"type":"doc","external_id":"doc:1","source_id":"eval","content":"alpha beta gamma"}',
                '{"type":"query","query":"alpha","relevant_external_ids":["doc:1"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    r = CliRunner().invoke(
        cli,
        ["eval", "--dataset", str(ds), "--k", "1", "--json-out", str(json_out)],
    )

    assert r.exit_code == 0, r.output
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "dataset_id",
        "retrieval_mode",
        "reranker_enabled",
        "k",
        "queries",
        "metrics",
    }
    assert set(payload["metrics"].keys()) == {"nDCG@1", "MAP@1", "MRR@1", "P@1", "Recall@1"}
