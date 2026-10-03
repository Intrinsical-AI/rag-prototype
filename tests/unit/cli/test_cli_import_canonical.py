from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from support.canonical import repogpt_payload

from local_rag_backend.cli import cli
from local_rag_backend.cli_commands.docs import docs_import_canonical as import_cmd_module
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import get_settings

settings = get_settings()


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
    assert {
        doc.external_id
        for doc in SqlDocumentStorage(session_factory=in_memory_sqlite).get_all_documents()
    } == {"doc-2"}


def test_read_payload_rejects_non_object_json(tmp_path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('["not-an-object"]', encoding="utf-8")

    try:
        import_cmd_module._read_payload(payload)
    except ValueError as exc:
        assert "--json must contain a JSON object payload" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_cli_import_rejects_repogpt_code_units_v3(tmp_path) -> None:
    payload = tmp_path / "payload.json"
    invalid = repogpt_payload()
    invalid["schema_version"] = "3"
    payload.write_text(json.dumps(invalid), encoding="utf-8")

    result = CliRunner().invoke(cli, ["import-canonical", "--json", str(payload)])

    assert result.exit_code == 1
    assert "schema_version='5'" in result.output


@pytest.mark.parametrize(
    ("invalid_fields", "error_field"),
    [
        ({"documents": "bad-documents"}, "documents"),
        ({"documents": ["bad-item"]}, "documents.0"),
        ({"replace_scope": "true"}, "replace_scope"),
    ],
)
def test_cli_import_rejects_invalid_payload(tmp_path, invalid_fields, error_field) -> None:
    payload = tmp_path / "payload.json"
    raw = {
        "scope": "repo",
        "snapshot_id": "snap",
        "documents": [{"external_id": "doc-1", "content": "hello"}],
    }
    raw.update(invalid_fields)
    payload.write_text(json.dumps(raw), encoding="utf-8")

    result = CliRunner().invoke(cli, ["import-canonical", "--json", str(payload)])

    assert result.exit_code == 1
    assert error_field in result.output


def test_cli_import_canonical_honors_payload_replace_scope_false(
    in_memory_sqlite, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    first = tmp_path / "snap1.json"
    first.write_text(
        json.dumps(
            {
                "scope": "repogpt:demo",
                "snapshot_id": "snap-1",
                "replace_scope": True,
                "documents": [
                    {"external_id": "doc-1", "content": "alpha"},
                    {"external_id": "doc-2", "content": "beta"},
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
                "replace_scope": False,
                "documents": [
                    {"external_id": "doc-2", "content": "beta-v2"},
                ],
            }
        ),
        encoding="utf-8",
    )

    r1 = CliRunner().invoke(cli, ["import-canonical", "--json", str(first)])
    assert r1.exit_code == 0, r1.output

    r2 = CliRunner().invoke(cli, ["import-canonical", "--json", str(second)])
    assert r2.exit_code == 0, r2.output
    assert "deleted_sql=0" in r2.output
    assert {
        doc.external_id
        for doc in SqlDocumentStorage(session_factory=in_memory_sqlite).get_all_documents()
    } == {
        "doc-1",
        "doc-2",
    }


def test_cli_import_canonical_upsert_only_overrides_payload_replace_scope_true(
    in_memory_sqlite, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    first = tmp_path / "snap1.json"
    first.write_text(
        json.dumps(
            {
                "scope": "repogpt:demo",
                "snapshot_id": "snap-1",
                "replace_scope": True,
                "documents": [
                    {"external_id": "doc-1", "content": "alpha"},
                    {"external_id": "doc-2", "content": "beta"},
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
                "replace_scope": True,
                "documents": [
                    {"external_id": "doc-2", "content": "beta-v2"},
                ],
            }
        ),
        encoding="utf-8",
    )

    r1 = CliRunner().invoke(cli, ["import-canonical", "--json", str(first)])
    assert r1.exit_code == 0, r1.output

    r2 = CliRunner().invoke(
        cli,
        ["import-canonical", "--upsert-only", "--json", str(second)],
    )
    assert r2.exit_code == 0, r2.output
    assert "deleted_sql=0" in r2.output
    assert {
        doc.external_id
        for doc in SqlDocumentStorage(session_factory=in_memory_sqlite).get_all_documents()
    } == {
        "doc-1",
        "doc-2",
    }


def test_cli_import_canonical_replace_scope_overrides_payload_replace_scope_false(
    in_memory_sqlite, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    first = tmp_path / "snap1.json"
    first.write_text(
        json.dumps(
            {
                "scope": "repogpt:demo",
                "snapshot_id": "snap-1",
                "replace_scope": True,
                "documents": [
                    {"external_id": "doc-1", "content": "alpha"},
                    {"external_id": "doc-2", "content": "beta"},
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
                "replace_scope": False,
                "documents": [
                    {"external_id": "doc-2", "content": "beta-v2"},
                ],
            }
        ),
        encoding="utf-8",
    )

    r1 = CliRunner().invoke(cli, ["import-canonical", "--json", str(first)])
    assert r1.exit_code == 0, r1.output

    r2 = CliRunner().invoke(
        cli,
        ["import-canonical", "--replace-scope", "--json", str(second)],
    )
    assert r2.exit_code == 0, r2.output
    assert "deleted_sql=1" in r2.output
    assert {
        doc.external_id
        for doc in SqlDocumentStorage(session_factory=in_memory_sqlite).get_all_documents()
    } == {"doc-2"}


def test_cli_import_canonical_reports_invalid_json(tmp_path) -> None:
    payload = tmp_path / "invalid.json"
    payload.write_text("{not-json", encoding="utf-8")

    result = CliRunner().invoke(cli, ["import-canonical", "--json", str(payload)])

    assert result.exit_code == 1
    assert "[ERROR] Error importing canonical documents:" in result.output


def test_cli_import_canonical_rejects_blank_document_without_deleting_scope(
    in_memory_sqlite, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    first = tmp_path / "snap1.json"
    first.write_text(
        json.dumps(
            {
                "scope": "repogpt:demo",
                "snapshot_id": "snap-1",
                "replace_scope": True,
                "documents": [
                    {"external_id": "doc-1", "content": "alpha"},
                ],
            }
        ),
        encoding="utf-8",
    )
    invalid = tmp_path / "snap2-invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "scope": "repogpt:demo",
                "snapshot_id": "snap-2",
                "replace_scope": True,
                "documents": [
                    {"external_id": "doc-1", "content": "   "},
                ],
            }
        ),
        encoding="utf-8",
    )

    r1 = CliRunner().invoke(cli, ["import-canonical", "--json", str(first)])
    assert r1.exit_code == 0, r1.output

    r2 = CliRunner().invoke(cli, ["import-canonical", "--json", str(invalid)])
    assert r2.exit_code == 1
    assert "content must not be blank" in r2.output
    assert {
        doc.external_id
        for doc in SqlDocumentStorage(session_factory=in_memory_sqlite).get_all_documents()
    } == {"doc-1"}
