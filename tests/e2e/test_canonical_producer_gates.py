from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.mcp_server import tool_import_canonical
from local_rag_backend.settings import get_settings


def _partial_payload() -> dict:
    payload = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "tr3v0r_canonical_v2.json").read_text()
    )
    payload["replace_scope"] = False
    payload["completion"].update(
        complete=False,
        failures=["No complete-run manifest supplied"],
        run_manifest=None,
        run_manifest_sha256=None,
    )
    return payload


@pytest.mark.parametrize("transport", ["cli", "mcp", "http"])
async def test_partial_rf_replacement_is_refused_before_mutation(
    transport, asgi_client, in_memory_sqlite, monkeypatch, tmp_path
):
    monkeypatch.setattr(get_settings(), "retrieval_mode", "sparse")
    payload = _partial_payload()
    seeded = await asgi_client.post(
        "/api/docs/import-canonical",
        json={
            "scope": payload["scope"],
            "snapshot_id": "seed",
            "documents": [
                {"external_id": "existing-owner-document", "content": "preserve this document"}
            ],
        },
    )
    assert seeded.status_code == 200
    if transport == "cli":
        path = tmp_path / "partial.json"
        path.write_text(json.dumps(payload))
        result = CliRunner().invoke(
            cli, ["import-canonical", "--replace-scope", "--json", str(path)]
        )
        assert result.exit_code == 1
        assert "Partial canonical" in result.output
    elif transport == "mcp":
        with pytest.raises(ValueError, match="Partial canonical"):
            tool_import_canonical(payload, replace_scope_override=True)
    else:
        payload["replace_scope"] = True
        result = await asgi_client.post("/api/docs/import-canonical", json=payload)
        assert result.status_code == 422
        assert "Partial canonical" in result.text
    documents = list(SqlDocumentStorage(session_factory=in_memory_sqlite).get_all_documents())
    assert [document.external_id for document in documents] == ["existing-owner-document"]


async def test_partial_rf_explicit_upsert_preserves_absent_documents(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(get_settings(), "retrieval_mode", "sparse")
    payload = _partial_payload()
    seeded = await asgi_client.post(
        "/api/docs/import-canonical",
        json={
            "scope": payload["scope"],
            "snapshot_id": "seed",
            "documents": [
                {"external_id": "existing-owner-document", "content": "preserve this document"}
            ],
        },
    )
    assert seeded.status_code == 200
    imported = await asgi_client.post("/api/docs/import-canonical", json=payload)
    assert imported.status_code == 200
    assert imported.json()["replace_scope"] is False
    assert imported.json()["deleted_sql"] == 0
    assert len(list(SqlDocumentStorage(session_factory=in_memory_sqlite).get_all_documents())) == 5
