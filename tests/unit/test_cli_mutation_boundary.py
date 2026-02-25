from __future__ import annotations

import json

from click.testing import CliRunner

from local_rag_backend.app.contracts.results import MutationSummary
from local_rag_backend.cli import cli
from local_rag_backend.cli_commands import docs_ingest, docs_mutate
from local_rag_backend.settings import settings


def test_cli_mutate_docs_calls_coordinator_execute_once(in_memory_sqlite, tmp_path, monkeypatch):
    calls = 0

    class FakeCoordinator:
        def __init__(self, *, settings_obj, ports):
            pass

        def execute(self, intent):
            nonlocal calls
            calls += 1
            return MutationSummary(
                op_id=str(intent.op_id or "fake-op"),
                inserted=len(list(intent.upserts)),
            )

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(docs_mutate, "MutationCoordinator", FakeCoordinator, raising=True)

    payload = tmp_path / "mutate.json"
    payload.write_text(
        json.dumps({"upserts": [{"external_id": "doc-1", "content": "hello"}]}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(cli, ["mutate-docs", "--json", str(payload)])
    assert result.exit_code == 0, result.output
    assert calls == 1


def test_cli_ingest_calls_coordinator_execute_once(in_memory_sqlite, tmp_path, monkeypatch):
    calls = 0

    class FakeCoordinator:
        def __init__(self, *, settings_obj, ports):
            pass

        def execute(self, intent):
            nonlocal calls
            calls += 1
            return MutationSummary(
                op_id=str(intent.op_id or "fake-op"),
                inserted=len(list(intent.upserts)),
                deleted_sql=len(list(intent.delete_ids)),
            )

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(docs_ingest, "MutationCoordinator", FakeCoordinator, raising=True)

    root = tmp_path / "ingest"
    root.mkdir()
    (root / "a.txt").write_text("hello ingest", encoding="utf-8")

    result = CliRunner().invoke(cli, ["ingest", str(root), "--no-magic"])
    assert result.exit_code == 0, result.output
    assert calls == 1
