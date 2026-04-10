from __future__ import annotations

import pytest

from local_rag_backend.core.services.docs_mutation_transport import (
    build_docs_mutation_intent_from_raw,
    validate_docs_mutation_payload,
)


def test_validate_docs_mutation_payload_normalizes_delete_lists() -> None:
    payload = validate_docs_mutation_payload(
        {
            "upserts": [{"external_id": "doc-1", "content": "hello"}],
            "delete_ids": [" id-1 ", "id-1", "", "id-2"],
            "delete_external_ids": [" ext-1 ", "ext-1", " ", "ext-2"],
        }
    )
    assert payload.delete_ids == ["id-1", "id-2"]
    assert payload.delete_external_ids == ["ext-1", "ext-2"]


def test_validate_docs_mutation_payload_rejects_empty_operation() -> None:
    with pytest.raises(ValueError, match="requires at least one operation"):
        validate_docs_mutation_payload({})


def test_validate_docs_mutation_payload_rejects_conflicting_external_ids() -> None:
    with pytest.raises(ValueError, match="cannot target the same external_id"):
        validate_docs_mutation_payload(
            {
                "upserts": [{"external_id": "doc-1", "content": "hello"}],
                "delete_external_ids": ["doc-1"],
            }
        )


def test_build_docs_mutation_intent_from_raw_preserves_source() -> None:
    intent = build_docs_mutation_intent_from_raw(
        {
            "op_id": " op-1 ",
            "upserts": [{"external_id": "doc-1", "content": "hello"}],
        },
        source="cli:docs:mutate",
    )
    assert intent.source == "cli:docs:mutate"
    assert intent.op_id == "op-1"
    assert len(intent.upserts) == 1
