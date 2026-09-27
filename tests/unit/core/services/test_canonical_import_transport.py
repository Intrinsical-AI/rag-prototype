from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError
from support.canonical import repogpt_payload
from support.repogpt_fixture import CANONICAL_FIXTURE

from local_rag_backend.core.services.canonical_import_transport import (
    build_canonical_import_request_input_from_raw,
    validate_canonical_import_payload,
)


def test_validate_and_project_repogpt_v4_retains_unit_fields_and_extension_metadata():
    raw = repogpt_payload()
    raw["documents"][0]["metadata"]["attributes"] = {"returns": "int"}
    before = copy.deepcopy(raw)
    payload = validate_canonical_import_payload(raw)
    request = build_canonical_import_request_input_from_raw(payload.model_dump(), source="test")
    document = request.documents[0]
    assert request.scope == raw["scope"]
    assert request.snapshot_id == raw["snapshot_id"]
    assert request.replace_scope is True
    assert document.source_id == raw["documents"][0]["source_id"]
    expected = {
        **raw["documents"][0]["metadata"],
        **{
            key: value
            for key, value in raw["documents"][0].items()
            if key not in {"external_id", "source_id", "content", "metadata"}
        },
    }
    assert document.metadata == expected
    assert raw == before


def test_real_versioned_producer_fixture_projects_every_document():
    raw = json.loads(CANONICAL_FIXTURE.read_text())
    request = build_canonical_import_request_input_from_raw(raw, source="test")
    assert len(request.documents) == raw["stats"]["emitted_documents"]
    for original, adapted in zip(raw["documents"], request.documents, strict=True):
        for key in ("path", "unit_type", "qualified_name", "repo_key", "content_hash"):
            assert adapted.metadata[key] == original[key]
        assert adapted.metadata["file"] == original["metadata"]["file"]


def test_generic_canonical_document_keeps_its_metadata_without_producer_requirements():
    metadata = {"path": "plain.txt", "repo_key": "custom", "extra": {"values": ["a"]}}
    raw = {
        "scope": "plain",
        "snapshot_id": "s1",
        "stats": {"failed_files": 0},
        "documents": [{"external_id": "doc", "content": "text", "metadata": metadata}],
    }
    request = build_canonical_import_request_input_from_raw(raw, source="test")
    assert request.documents[0].metadata == metadata


@pytest.mark.parametrize(
    "field", ["path", "unit_type", "repo_key", "content_hash", "qualified_name"]
)
def test_repo_gpt_unit_fields_must_be_at_document_level(field):
    raw = repogpt_payload()
    document = raw["documents"][0]
    document["metadata"][field] = document.pop(field)
    with pytest.raises(ValueError):
        validate_canonical_import_payload(raw)


@pytest.mark.parametrize("field", ["repo_key", "scope", "snapshot_id"])
def test_repo_gpt_document_identity_must_match_envelope(field):
    raw = repogpt_payload()
    raw["documents"][0][field] = "different"
    with pytest.raises(ValueError, match=f"must match payload.{field}"):
        validate_canonical_import_payload(raw)


def test_validate_canonical_import_payload_rejects_repogpt_v3():
    raw = repogpt_payload()
    raw["schema_version"] = "3"
    with pytest.raises(ValueError, match="schema_version='4'"):
        validate_canonical_import_payload(raw)


def test_validate_canonical_import_payload_rejects_non_boolean_replace_scope():
    with pytest.raises(ValidationError, match="replace_scope"):
        validate_canonical_import_payload(
            {
                "scope": "repo:demo",
                "snapshot_id": "snap-1",
                "replace_scope": "true",
                "documents": [{"external_id": "doc-1", "content": "alpha"}],
            }
        )
