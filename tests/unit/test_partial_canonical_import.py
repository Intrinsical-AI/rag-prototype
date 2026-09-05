from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.core.services.canonical_import_transport import (
    build_canonical_import_request_input,
    validate_canonical_import_payload,
)
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.mcp_server import tool_import_canonical
from local_rag_backend.settings import settings


def _payload(*, failed_files=1, failures=None, replace_scope=True):
    return {
        "kind": "code-units",
        "schema_version": "4",
        "repo_key": "demo",
        "scope": "repogpt:demo",
        "snapshot_id": "new",
        "replace_scope": replace_scope,
        "stats": {"failed_files": failed_files},
        "failures": failures or [],
        "documents": [
            {
                "external_id": "new-doc",
                "content": "new content",
                "metadata": {
                    "path": "app.py",
                    "unit_type": "module",
                    "repo_key": "demo",
                    "content_hash": "new-hash",
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "evidence", [{"failed_files": 1}, {"failed_files": 0, "failures": [{"path": "bad.py"}]}]
)
def test_partial_evidence_survives_validation_and_effective_override(evidence):
    parsed = validate_canonical_import_payload(_payload(**evidence))
    parsed = validate_canonical_import_payload(parsed.model_dump())
    with pytest.raises(ValueError, match="Partial canonical exports"):
        build_canonical_import_request_input(parsed, source="test")
    assert not build_canonical_import_request_input(
        parsed, source="test", replace_scope_override=False
    ).replace_scope
    parsed.replace_scope = False
    with pytest.raises(ValueError, match="Partial canonical exports"):
        build_canonical_import_request_input(parsed, source="test", replace_scope_override=True)


@pytest.mark.parametrize("transport", ["http", "cli", "mcp"])
async def test_partial_rejected_without_writes_and_explicit_upsert_preserves_absent(
    transport, asgi_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")
    repo = SqlDocumentStorage()
    repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id="absent-doc", content="keep me", scope="repogpt:demo")]
    )
    payload = _payload()

    async def invoke(*, upsert_only):
        if transport == "http":
            response = await asgi_client.post(
                "/api/docs/import-canonical", json={**payload, "replace_scope": not upsert_only}
            )
            return response.status_code == 200, response.text
        if transport == "cli":
            path = tmp_path / "partial.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            args = ["import-canonical", "--json", str(path)]
            if upsert_only:
                args.append("--upsert-only")
            result = CliRunner().invoke(cli, args)
            return result.exit_code == 0, result.output
        try:
            tool_import_canonical(payload, replace_scope_override=False if upsert_only else None)
            return True, ""
        except ValueError as exc:
            return False, str(exc)

    ok, detail = await invoke(upsert_only=False)
    assert not ok
    assert "Partial canonical exports" in detail
    assert [(doc.external_id, doc.content) for doc in repo.get_all_documents()] == [
        ("absent-doc", "keep me")
    ]
    ok, detail = await invoke(upsert_only=True)
    assert ok, detail
    assert {doc.external_id for doc in repo.get_all_documents()} == {"absent-doc", "new-doc"}
    assert not repo.get_tombstoned_external_ids(["absent-doc"])
