from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError
from support.canonical import repogpt_empty_module_payload, repogpt_payload
from support.repogpt_fixture import CANONICAL_FIXTURE

from local_rag_backend.core.services.canonical_import_transport import (
    build_canonical_import_request_input_from_raw,
    validate_canonical_import_payload,
)


def test_validate_and_project_repogpt_v5_retains_unit_fields_and_extension_metadata():
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
    indexable = [
        document
        for document in raw["documents"]
        if document["unit_type"] != "module" or document["content"].strip()
    ]
    assert len(request.documents) == len(indexable)
    for original, adapted in zip(indexable, request.documents, strict=True):
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
    with pytest.raises(ValueError, match="schema_version='5'"):
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


def test_v5_filters_blank_structural_module_after_full_validation():
    raw = repogpt_payload()
    module = repogpt_empty_module_payload()["documents"][0]
    raw["documents"].insert(0, module)
    raw["stats"]["emitted_documents"] = 2
    before = copy.deepcopy(raw)

    request = build_canonical_import_request_input_from_raw(raw, source="test")

    assert [document.external_id for document in request.documents] == [
        raw["documents"][1]["external_id"]
    ]
    assert request.allow_empty is True
    assert raw == before


def test_v5_only_blank_module_produces_empty_import_request():
    raw = repogpt_empty_module_payload()

    request = build_canonical_import_request_input_from_raw(raw, source="test")

    assert request.documents == ()
    assert request.allow_empty is True


@pytest.mark.parametrize("invalid", ["missing_ranges", "bad_hash", "blank_symbol"])
def test_v5_rejects_invalid_document_before_filtering(invalid: str):
    raw = repogpt_empty_module_payload()
    module = raw["documents"][0]
    if invalid == "missing_ranges":
        del module["content_ranges"]
    elif invalid == "bad_hash":
        module["content_hash"] = "0" * 64
    else:
        module["unit_type"] = "function"
    with pytest.raises(ValueError):
        validate_canonical_import_payload(raw)


def test_v5_rejects_raw_empty_documents_before_scope_sync():
    raw = repogpt_payload()
    raw["documents"] = []
    raw["stats"]["emitted_documents"] = 0
    with pytest.raises(ValueError, match="non-empty documents list"):
        build_canonical_import_request_input_from_raw(raw, source="test")


def test_v5_rejects_mismatched_emitted_document_count():
    raw = repogpt_empty_module_payload()
    raw["stats"]["emitted_documents"] = 0
    with pytest.raises(ValueError, match="emitted_documents"):
        validate_canonical_import_payload(raw)


def test_v5_overlong_recoverable_document_rejected_before_import():
    raw = repogpt_payload(content="x" * 20_001)
    with pytest.raises(ValidationError, match="content"):
        build_canonical_import_request_input_from_raw(raw, source="test")


def test_generic_empty_import_remains_invalid():
    raw = {"scope": "generic", "snapshot_id": "one", "documents": []}
    with pytest.raises(ValueError, match="non-empty documents list"):
        validate_canonical_import_payload(raw)
