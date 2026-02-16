# tests/unit/test_cli_eval.py

from pathlib import Path

from click.testing import CliRunner

from local_rag_backend.cli import cli


def test_rag_eval_fails_below_threshold(tmp_path: Path) -> None:
    # Minimal dataset where the relevant_external_ids don't exist => hit_rate=0.
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        "\n".join(
            [
                '{"type":"meta","dataset_id":"x","schema_version":1,"created_at":"2026-02-16"}',
                '{"type":"doc","external_id":"doc:1","source_id":"eval","content":"alpha beta gamma"}',
                '{"type":"query","query":"alpha","relevant_external_ids":["doc:missing"]}',
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
            "--fail-below-hit-rate",
            "0.5",
        ],
    )
    assert r.exit_code == 1, r.output
    assert "Eval regression" in r.output
