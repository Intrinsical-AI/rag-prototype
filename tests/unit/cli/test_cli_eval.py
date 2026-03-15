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
