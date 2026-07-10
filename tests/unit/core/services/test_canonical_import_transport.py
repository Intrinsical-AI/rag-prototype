from __future__ import annotations

from pydantic import ValidationError

from local_rag_backend.core.services.canonical_import_transport import (
    build_canonical_import_request_input_from_raw,
    validate_canonical_import_payload,
)


def test_validate_canonical_import_payload_accepts_repogpt_v4() -> None:
    payload = validate_canonical_import_payload(
        {
            "schema_version": "4",
            "kind": "code-units",
            "repo_key": "demo",
            "scope": "repogpt:demo",
            "snapshot_id": "snap-1",
            "replace_scope": True,
            "documents": [
                {
                    "external_id": "repogpt:demo:src/app.py:function:helper",
                    "source_id": "repogpt:demo:file:src/app.py",
                    "scope": "repogpt:demo",
                    "snapshot_id": "snap-1",
                    "path": "src/app.py",
                    "unit_type": "function",
                    "repo_key": "demo",
                    "content_hash": "abc123",
                    "content": "def helper():\n    return 1\n",
                    "metadata": {
                        "scope": "repogpt:demo",
                        "snapshot_id": "snap-1",
                        "path": "src/app.py",
                        "unit_type": "function",
                        "repo_key": "demo",
                        "content_hash": "abc123",
                    },
                }
            ],
        }
    )

    request = build_canonical_import_request_input_from_raw(
        payload.model_dump(),
        source="test",
    )
    assert request.scope == "repogpt:demo"
    assert request.snapshot_id == "snap-1"
    assert request.replace_scope is True
    assert len(request.documents) == 1


def test_validate_canonical_import_payload_rejects_repogpt_v3() -> None:
    try:
        validate_canonical_import_payload(
            {
                "schema_version": "3",
                "kind": "code-units",
                "repo_key": "demo",
                "scope": "repogpt:demo",
                "snapshot_id": "snap-1",
                "documents": [
                    {
                        "external_id": "doc-1",
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
        )
    except ValueError as exc:
        assert "schema_version='4'" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_canonical_import_payload_rejects_non_boolean_replace_scope() -> None:
    try:
        validate_canonical_import_payload(
            {
                "scope": "repo:demo",
                "snapshot_id": "snap-1",
                "replace_scope": "true",
                "documents": [{"external_id": "doc-1", "content": "alpha"}],
            }
        )
    except ValidationError as exc:
        assert "replace_scope" in str(exc)
    else:
        raise AssertionError("expected ValidationError")
