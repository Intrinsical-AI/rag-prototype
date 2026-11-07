from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_rag_backend.core.services.canonical_import_transport import (
    build_canonical_import_request_input_from_raw,
)


def payload():
    return {
        "scope": "closure-only",
        "snapshot_id": "synthetic",
        "documents": [{"external_id": "one", "content": "synthetic test"}],
    }


def test_generic_import_omission_is_upsert_only():
    assert (
        build_canonical_import_request_input_from_raw(payload(), source="test").replace_scope
        is False
    )


@pytest.mark.parametrize("error_count", ["3", None, True, False, 0.0, -1])
def test_invalid_error_count_refused_before_request_assembly(error_count):
    raw = {**payload(), "stats": {"error_count": error_count}}
    with pytest.raises(ValidationError, match="error_count"):
        build_canonical_import_request_input_from_raw(raw, source="test")


@pytest.mark.parametrize("override", [None, True])
def test_partial_generic_import_cannot_replace_with_override(override):
    raw = {**payload(), "replace_scope": True, "stats": {"error_count": 1}}
    with pytest.raises(ValueError, match="Partial canonical"):
        build_canonical_import_request_input_from_raw(
            raw, source="test", replace_scope_override=override
        )
