from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from support.canonical import repogpt_empty_module_payload, repogpt_failure, repogpt_payload

from local_rag_backend.cli import cli
from local_rag_backend.core.services.canonical_import_transport import (
    build_canonical_import_request_input,
    validate_canonical_import_payload,
)
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.mcp_server import tool_import_canonical
from local_rag_backend.settings import get_settings

settings = get_settings()


def _payload(*, failed_files=1, failures=None, replace_scope=True):
    return repogpt_payload(
        external_id="new-doc",
        content="new content",
        failed_files=failed_files,
        failures=failures,
        replace_scope=replace_scope,
    )


@pytest.mark.parametrize(
    "evidence", [{"failed_files": 1}, {"failed_files": 0, "failures": [repogpt_failure("bad.py")]}]
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
    transport, asgi_client, in_memory_sqlite, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
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


@pytest.mark.parametrize("transport", ["http", "cli", "mcp"])
@pytest.mark.parametrize("failed_files", [0, 1])
@pytest.mark.parametrize("upsert_only", [False, True])
async def test_empty_v5_snapshot_sync_or_no_op_by_effective_replace_scope(
    transport, failed_files, upsert_only, asgi_client, in_memory_sqlite, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id="last-doc", content="keep last document", scope="repogpt:demo")]
    )
    before = list(repo.get_all_documents())
    payload = repogpt_empty_module_payload(failed_files=failed_files, replace_scope=not upsert_only)
    if failed_files:
        payload["failures"] = [repogpt_failure()]
    if transport == "http":
        response = await asgi_client.post("/api/docs/import-canonical", json=payload)
        accepted = response.status_code == 200
        detail = response.text
    elif transport == "cli":
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        args = ["import-canonical", "--json", str(path)]
        if upsert_only:
            args.append("--upsert-only")
        result = CliRunner().invoke(cli, args)
        accepted = result.exit_code == 0
        detail = result.output
    else:
        try:
            tool_import_canonical(payload, replace_scope_override=False if upsert_only else None)
            accepted, detail = True, ""
        except ValueError as exc:
            accepted, detail = False, str(exc)

    if failed_files and not upsert_only:
        assert not accepted
        assert "Partial canonical exports" in detail
        assert list(repo.get_all_documents()) == before
    elif upsert_only:
        assert accepted, detail
        assert list(repo.get_all_documents()) == before
    else:
        assert accepted, detail
        assert list(repo.get_all_documents()) == []
