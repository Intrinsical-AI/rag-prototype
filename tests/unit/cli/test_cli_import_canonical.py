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


def test_read_payload_accepts_repogpt_code_units_v4(tmp_path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps(
            {
                "schema_version": "4",
                "kind": "code-units",
                "repo_key": "demo",
                "scope": "repogpt:demo",
                "snapshot_id": "demo-snap",
                "replace_scope": True,
                "documents": [
                    {
                        "external_id": "repogpt:demo:src/app.py:function:helper",
                        "source_id": "repogpt:demo:file:src/app.py",
                        "scope": "repogpt:demo",
                        "snapshot_id": "demo-snap",
                        "path": "src/app.py",
                        "unit_type": "function",
                        "repo_key": "demo",
                        "content_hash": "abc123",
                        "content": "def helper():\n    return 1\n",
                        "metadata": {
                            "scope": "repogpt:demo",
                            "snapshot_id": "demo-snap",
                            "path": "src/app.py",
                            "unit_type": "function",
                            "repo_key": "demo",
                            "content_hash": "abc123",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = import_cmd_module._read_payload(payload)
    assert loaded["schema_version"] == "4"
    assert loaded["kind"] == "code-units"


def test_read_payload_rejects_repogpt_code_units_v3(tmp_path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps(
            {
                "schema_version": "3",
                "kind": "code-units",
                "repo_key": "demo",
                "scope": "repogpt:demo",
                "snapshot_id": "demo-snap",
                "documents": [
                    {
                        "external_id": "repogpt:demo:1",
                        "content": "def helper():\n    return 1\n",
                        "metadata": {
                            "path": "src/app.py",
                            "unit_type": "function",
                            "repo_key": "demo",
                            "content_hash": "abc123",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        import_cmd_module._read_payload(payload)
    except ValueError as exc:
        assert "schema_version='4'" in str(exc)
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


def test_resolve_replace_scope_defaults_to_true_when_payload_is_missing() -> None:
    assert import_cmd_module._resolve_replace_scope({}, replace_scope_override=None) is True


def test_resolve_replace_scope_uses_payload_when_no_flag_override() -> None:
    assert (
        import_cmd_module._resolve_replace_scope(
            {"replace_scope": False},
            replace_scope_override=None,
        )
        is False
    )
    assert (
        import_cmd_module._resolve_replace_scope(
            {"replace_scope": True},
            replace_scope_override=None,
        )
        is True
    )


def test_resolve_replace_scope_explicit_flags_override_payload() -> None:
    assert (
        import_cmd_module._resolve_replace_scope(
            {"replace_scope": False},
            replace_scope_override=True,
        )
        is True
    )
    assert (
        import_cmd_module._resolve_replace_scope(
            {"replace_scope": True},
            replace_scope_override=False,
        )
        is False
    )


def test_resolve_replace_scope_rejects_non_boolean_payload_value() -> None:
    try:
        import_cmd_module._resolve_replace_scope(
            {"replace_scope": "true"},
            replace_scope_override=None,
        )
    except ValueError as exc:
        assert "payload.replace_scope must be a boolean" in str(exc)
    else:
        raise AssertionError("expected ValueError")


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
    assert {doc.external_id for doc in SqlDocumentStorage().get_all_documents()} == {
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
    assert {doc.external_id for doc in SqlDocumentStorage().get_all_documents()} == {
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
    assert {doc.external_id for doc in SqlDocumentStorage().get_all_documents()} == {"doc-2"}


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
    assert {doc.external_id for doc in SqlDocumentStorage().get_all_documents()} == {"doc-1"}
