from __future__ import annotations

import json

from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.cli_commands.docs import docs_import_canonical as import_cmd_module
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


def test_cli_import_canonical_syncs_scope(in_memory_sqlite, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    first = tmp_path / "snap1.json"
    first.write_text(
        json.dumps(
            {
                "scope": "repogpt:demo",
                "snapshot_id": "snap-1",
                "documents": [
                    {
                        "external_id": "doc-1",
                        "source_id": "file:a.py",
                        "content": "def alpha():\n    return 1\n",
                        "metadata": {"path": "a.py"},
                    },
                    {
                        "external_id": "doc-2",
                        "source_id": "file:b.py",
                        "content": "def beta():\n    return 2\n",
                        "metadata": {"path": "b.py"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    second = tmp_path / "snap2.json"
    second.write_text(
        json.dumps(
            {
                "scope": "repogpt:demo",
                "snapshot_id": "snap-2",
                "documents": [
                    {
                        "external_id": "doc-2",
                        "source_id": "file:b.py",
                        "content": "def beta():\n    return 20\n",
                        "metadata": {"path": "b.py"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    r1 = CliRunner().invoke(cli, ["import-canonical", "--json", str(first)])
    assert r1.exit_code == 0, r1.output
    assert "inserted=2" in r1.output

    r2 = CliRunner().invoke(cli, ["import-canonical", "--json", str(second)])
    assert r2.exit_code == 0, r2.output
    assert "updated=1" in r2.output
    assert "deleted_sql=1" in r2.output
    assert {doc.external_id for doc in SqlDocumentStorage().get_all_documents()} == {"doc-2"}


def test_read_payload_rejects_non_object_json(tmp_path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('["not-an-object"]', encoding="utf-8")

    try:
        import_cmd_module._read_payload(payload)
    except ValueError as exc:
        assert "--json must contain a JSON object payload" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_request_rejects_non_list_documents() -> None:
    try:
        import_cmd_module._build_request(
            {"scope": "repo", "snapshot_id": "snap", "documents": "bad-documents"},
            replace_scope=True,
        )
    except ValueError as exc:
        assert "payload.documents must be a list" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_request_rejects_non_object_document_items() -> None:
    try:
        import_cmd_module._build_request(
            {"scope": "repo", "snapshot_id": "snap", "documents": ["bad-item"]},
            replace_scope=True,
        )
    except ValueError as exc:
        assert "each payload.documents item must be an object" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_cli_import_canonical_reports_invalid_json(tmp_path) -> None:
    payload = tmp_path / "invalid.json"
    payload.write_text("{not-json", encoding="utf-8")

    result = CliRunner().invoke(cli, ["import-canonical", "--json", str(payload)])

    assert result.exit_code == 1
    assert "[ERROR] Error importing canonical documents:" in result.output
