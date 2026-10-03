"""Produce accepted SQLite query DTOs for the local Chromium regression smoke."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.composition.adapters import _RepoDocsReadPort
from local_rag_backend.core.services.canonical_import_transport import (
    validate_canonical_import_payload,
)
from local_rag_backend.http.schemas.docs import DocsMutateRequest
from local_rag_backend.http.schemas.shared import DocumentInDB
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage, base as db_base

PROBE_ID = (
    "probe<img src='/missing-probe' "
    "onerror=\"document.documentElement.dataset.domProbe='executed'\">"
)


def accepted_query() -> list[dict[str, Any]]:
    documents = [
        {"external_id": PROBE_ID, "content": "Preview: <b>literal HTML</b>"},
        {"external_id": "doc:0198c3f7-3789-7123-89ab-0123456789ab", "content": "UUID document"},
        {"external_id": "repogpt:demo:src/example.py:function:hello", "content": "Canonical code"},
    ]
    canonical = validate_canonical_import_payload(
        {"scope": "browser-smoke", "snapshot_id": "one", "documents": documents}
    )
    mutation = DocsMutateRequest.model_validate({"upserts": documents})
    assert mutation.upserts[0].external_id == PROBE_ID

    with TemporaryDirectory(prefix="rag-frontend-smoke-") as temp_dir:
        engine = create_engine(
            f"sqlite:///{Path(temp_dir) / 'app.db'}",
            connect_args={"check_same_thread": False},
        )
        try:
            db_base.ensure_sqlite_schema_current(engine_to_use=engine)
            sessions = sessionmaker(bind=engine, autocommit=False, autoflush=False)
            repo = SqlDocumentStorage(session_factory=sessions)
            repo.upsert_documents_by_external_id(
                [
                    repo.UpsertDoc(
                        external_id=d.external_id,
                        content=d.content,
                        scope=canonical.scope,
                        snapshot_id=canonical.snapshot_id,
                    )
                    for d in canonical.documents
                ]
            )
            reader = _RepoDocsReadPort(doc_repo_factory=lambda: repo)
            output = [
                DocumentInDB.model_validate(row).model_dump()
                for row in reader.query_docs(limit=100, offset=0, filters=())
            ]
        finally:
            engine.dispose()
    assert {doc["external_id"] for doc in output} == {d.external_id for d in canonical.documents}
    assert any(doc["external_id"] == PROBE_ID for doc in output)
    return output


if __name__ == "__main__":
    print(json.dumps(accepted_query()))
