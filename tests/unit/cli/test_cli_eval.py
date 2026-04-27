# tests/unit/test_cli_eval.py

import json
from pathlib import Path

from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.composition import factory
from local_rag_backend.settings import settings


class DummyEmbedder:
    dim = 2

    def embed(self, texts):
        out = []
        for text in texts:
            normalized = text.lower()
            if "auth" in normalized:
                out.append([1.0, 0.0])
            elif "sql" in normalized:
                out.append([0.0, 1.0])
            else:
                out.append([0.5, 0.5])
        return out


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


def test_rag_eval_writes_detailed_report_and_anomalies(tmp_path: Path) -> None:
    ds = tmp_path / "ds.jsonl"
    report_out = tmp_path / "report.json"
    anomalies_out = tmp_path / "anomalies.jsonl"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc:1","source_id":"eval","content":"alpha beta"}',
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
            "--k",
            "1",
            "--report-out",
            str(report_out),
            "--anomalies-out",
            str(anomalies_out),
        ],
    )

    assert r.exit_code == 0, r.output
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["type"] == "eval_report"
    assert report["per_query"][0]["ranked_docs"][0]["external_id"] == "doc:1"
    assert anomalies_out.read_text(encoding="utf-8") == ""


def test_rag_eval_dense_mode_uses_isolated_runtime_and_json_out(
    tmp_path: Path, monkeypatch
) -> None:
    ds = tmp_path / "dense.jsonl"
    json_out = tmp_path / "dense-result.json"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc-auth","source_id":"eval","content":"auth auth"}',
                '{"type":"doc","external_id":"doc-sql","source_id":"eval","content":"sql sql"}',
                '{"type":"query","query":"auth","relevant_external_ids":["doc-auth"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "vector_backend", "numpy", raising=False)
    monkeypatch.setattr(
        factory,
        "SentenceTransformerEmbedder",
        lambda model_name=None: DummyEmbedder(),
        raising=True,
    )
    factory.reset_app_context()

    r = CliRunner().invoke(
        cli,
        [
            "eval",
            "--dataset",
            str(ds),
            "--retrieval-mode",
            "dense",
            "--k",
            "1",
            "--candidate-k",
            "1",
            "--json-out",
            str(json_out),
        ],
    )

    assert r.exit_code == 0, r.output
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["retrieval_mode"] == "dense"
    assert payload["metrics"]["nDCG@1"] == 1.0
    factory.reset_app_context()


def test_rag_eval_dense_mode_reports_missing_embeddings_backend_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    ds = tmp_path / "dense.jsonl"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc-auth","source_id":"eval","content":"auth auth"}',
                '{"type":"query","query":"auth","relevant_external_ids":["doc-auth"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _st_fail(model_name=None):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(factory, "SentenceTransformerEmbedder", _st_fail, raising=True)
    factory.reset_app_context()

    r = CliRunner().invoke(
        cli,
        ["eval", "--dataset", str(ds), "--retrieval-mode", "dense", "--k", "1"],
    )

    assert r.exit_code == 1
    assert "Dense/hybrid retrieval requires an embeddings backend" in r.output
    factory.reset_app_context()


def test_rag_eval_rejects_incompatible_mode_specific_flags(tmp_path: Path) -> None:
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc:1","source_id":"eval","content":"alpha"}',
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
            "--candidate-k",
            "5",
        ],
    )

    assert r.exit_code == 1
    assert "--candidate-k is supported only with retrieval_mode=dense" in r.output


def test_rag_eval_compare_passes_and_writes_json(tmp_path: Path, monkeypatch) -> None:
    ds = tmp_path / "compare.jsonl"
    spec = tmp_path / "compare-spec.json"
    json_out = tmp_path / "compare-result.json"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc-auth","source_id":"eval","content":"auth auth"}',
                '{"type":"doc","external_id":"doc-sql","source_id":"eval","content":"sql sql"}',
                '{"type":"query","query":"auth","relevant_external_ids":["doc-auth"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec.write_text(
        json.dumps(
            {
                "k": 1,
                "baseline": {"retrieval_mode": "sparse"},
                "candidate": {"retrieval_mode": "dual", "dual_candidate_k": 1},
                "thresholds": {
                    "min_delta_ndcg": 0.0,
                    "min_delta_map": 0.0,
                    "min_delta_mrr": 0.0,
                    "max_regression_precision": 0.0,
                    "max_regression_recall": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "vector_backend", "numpy", raising=False)
    monkeypatch.setattr(
        factory,
        "SentenceTransformerEmbedder",
        lambda model_name=None: DummyEmbedder(),
        raising=True,
    )
    factory.reset_app_context()

    r = CliRunner().invoke(
        cli,
        [
            "eval-compare",
            "--dataset",
            str(ds),
            "--spec",
            str(spec),
            "--json-out",
            str(json_out),
        ],
    )

    assert r.exit_code == 0, r.output
    assert "BASELINE" in r.output
    assert "CANDIDATE" in r.output
    assert "DELTA" in r.output
    assert "PASS" in r.output
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"dataset_id", "k", "baseline", "candidate", "delta", "gate"}
    assert payload["gate"]["passed"] is True
    factory.reset_app_context()


def test_rag_eval_compare_fails_gate_with_exit_code_one(tmp_path: Path) -> None:
    ds = tmp_path / "compare.jsonl"
    spec = tmp_path / "compare-spec.json"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc:1","source_id":"eval","content":"zzz zzz"}',
                '{"type":"doc","external_id":"doc:2","source_id":"eval","content":"alpha alpha"}',
                '{"type":"query","query":"alpha","relevant_external_ids":["doc:1"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec.write_text(
        json.dumps(
            {
                "k": 1,
                "baseline": {"retrieval_mode": "sparse"},
                "candidate": {"retrieval_mode": "sparse", "reranker_enabled": True},
                "thresholds": {"min_delta_ndcg": 0.1},
            }
        ),
        encoding="utf-8",
    )

    r = CliRunner().invoke(
        cli,
        [
            "eval-compare",
            "--dataset",
            str(ds),
            "--spec",
            str(spec),
        ],
    )

    assert r.exit_code == 1
    assert "FAIL:" in r.output
    assert "nDCG@1" in r.output


def test_rag_eval_compare_reports_invalid_candidate_mode_config_with_exit_code_two(
    tmp_path: Path,
) -> None:
    ds = tmp_path / "compare.jsonl"
    spec = tmp_path / "compare-spec.json"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc:1","source_id":"eval","content":"alpha"}',
                '{"type":"query","query":"alpha","relevant_external_ids":["doc:1"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec.write_text(
        json.dumps(
            {
                "baseline": {"retrieval_mode": "sparse"},
                "candidate": {"retrieval_mode": "sparse", "candidate_k": 5},
            }
        ),
        encoding="utf-8",
    )

    r = CliRunner().invoke(
        cli,
        [
            "eval-compare",
            "--dataset",
            str(ds),
            "--spec",
            str(spec),
        ],
    )

    assert r.exit_code == 2
    assert "[ERROR] Error comparing eval runs:" in r.output


def test_rag_eval_batch_runs_multiple_specs_and_writes_json(tmp_path: Path, monkeypatch) -> None:
    ds = tmp_path / "batch.jsonl"
    specs = tmp_path / "specs.json"
    sparse_json = tmp_path / "sparse.json"
    hybrid_json = tmp_path / "hybrid.json"
    sparse_run = tmp_path / "sparse.jsonl"
    hybrid_run = tmp_path / "hybrid.jsonl"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc-auth","source_id":"eval","content":"auth auth"}',
                '{"type":"doc","external_id":"doc-sql","source_id":"eval","content":"sql sql"}',
                '{"type":"query","query":"auth","relevant_external_ids":["doc-auth"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    specs.write_text(
        json.dumps(
            [
                {
                    "name": "sparse-a",
                    "retrieval_mode": "sparse",
                    "k": 1,
                    "json_out": str(sparse_json),
                    "run_out": str(sparse_run),
                },
                {
                    "name": "hybrid-a",
                    "retrieval_mode": "hybrid",
                    "k": 1,
                    "hybrid_alpha": 0.5,
                    "json_out": str(hybrid_json),
                    "run_out": str(hybrid_run),
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "vector_backend", "numpy", raising=False)
    monkeypatch.setattr(
        factory,
        "SentenceTransformerEmbedder",
        lambda model_name=None: DummyEmbedder(),
        raising=True,
    )
    factory.reset_app_context()

    r = CliRunner().invoke(
        cli,
        ["eval-batch", "--dataset", str(ds), "--specs", str(specs)],
    )

    assert r.exit_code == 0, r.output
    assert "sparse-a:" in r.output
    assert "hybrid-a:" in r.output
    assert json.loads(sparse_json.read_text(encoding="utf-8"))["retrieval_mode"] == "sparse"
    assert json.loads(hybrid_json.read_text(encoding="utf-8"))["retrieval_mode"] == "hybrid"
    assert sparse_run.exists()
    assert hybrid_run.exists()
    factory.reset_app_context()


def test_rag_eval_batch_rejects_invalid_specs_shape(tmp_path: Path) -> None:
    ds = tmp_path / "batch.jsonl"
    specs = tmp_path / "specs.json"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1}',
                '{"type":"doc","external_id":"doc-auth","source_id":"eval","content":"auth auth"}',
                '{"type":"query","query":"auth","relevant_external_ids":["doc-auth"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    specs.write_text(json.dumps({"name": "bad"}), encoding="utf-8")

    r = CliRunner().invoke(cli, ["eval-batch", "--dataset", str(ds), "--specs", str(specs)])

    assert r.exit_code == 1
    assert "Batch specs file must be a JSON array of objects" in r.output
