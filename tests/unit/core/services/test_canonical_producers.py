"""Producer-generated fixtures and destructive-import evidence regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from local_rag_backend.core.services.canonical_import_transport import (
    build_canonical_import_request_input_from_raw,
    validate_canonical_import_payload,
)

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"


def _payload(name: str = "tr3v0r_canonical_v2.json") -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    "name", ["tr3v0r_canonical_v2.json", "chat_units_v4.json", "repogpt_code_units_v6.json"]
)
def test_real_producer_envelopes_reach_the_native_import_dto(name: str):
    raw = _payload(name)
    request = build_canonical_import_request_input_from_raw(raw, source="regression")
    assert request.scope == raw["scope"]
    assert request.documents


@pytest.mark.parametrize("kind", [None, "unknown", "code-units", "chat-units"])
def test_unknown_null_or_conflicting_producer_identity_is_rejected(kind):
    raw = _payload()
    raw["kind"] = kind
    with pytest.raises(ValueError):
        validate_canonical_import_payload(raw)


def test_genuine_marker_free_native_import_remains_supported():
    raw = {
        "scope": "native",
        "snapshot_id": "one",
        "documents": [{"external_id": "one", "content": "native data"}],
    }
    assert build_canonical_import_request_input_from_raw(raw, source="native").documents


def test_obsolete_chat_envelope_cannot_masquerade_as_native_generic():
    raw = _payload("chat_units_v4.json")
    raw.pop("kind")
    raw.pop("schema_version")
    raw["producer"]["format_version"] = "3"
    with pytest.raises(ValueError, match="obsolete"):
        validate_canonical_import_payload(raw)


def test_previous_tr3v0r_schema_is_rejected():
    raw = _payload()
    raw["schema_version"] = "1"
    with pytest.raises(ValueError):
        validate_canonical_import_payload(raw)


@pytest.mark.parametrize("override", [None, True])
def test_incomplete_tr3v0r_cannot_replace_even_with_override(override):
    raw = _payload()
    raw["replace_scope"] = False
    raw["completion"].update(
        complete=False,
        failures=["No complete-run manifest supplied"],
        run_manifest=None,
        run_manifest_sha256=None,
    )
    if override is None:
        assert (
            build_canonical_import_request_input_from_raw(raw, source="partial").replace_scope
            is False
        )
    else:
        with pytest.raises(ValueError, match="Partial canonical"):
            build_canonical_import_request_input_from_raw(
                raw, source="partial", replace_scope_override=override
            )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_manifest",
        "document_digest",
        "receipt_digest",
        "wrong_dataset",
        "bad_stages",
        "bad_counters",
        "unknown_records",
        "missing_artifact",
        "unsafe_path",
        "artifact_count",
        "dropped_document",
        "bool_counter",
    ],
)
def test_forged_or_incoherent_completeness_is_rejected(mutation: str):
    raw = _payload()
    completion = raw["completion"]
    manifest = completion["run_manifest"]
    if mutation == "missing_manifest":
        completion["run_manifest"] = None
    elif mutation == "document_digest":
        raw["documents"][0]["content"] += " changed"
    elif mutation == "receipt_digest":
        completion["run_manifest_sha256"] = "0" * 64
    elif mutation == "wrong_dataset":
        completion["dataset_key"] = "different"
    elif mutation == "bad_stages":
        manifest["completed_stages"].remove("detect")
    elif mutation == "bad_counters":
        manifest["parser_counts"]["read"] += 1
    elif mutation == "unknown_records":
        manifest["parser_counts"]["unknown"] = 1
        manifest["parser_counts"]["read"] += 1
    elif mutation == "missing_artifact":
        manifest["artifacts"].pop("wifi_mac")
    elif mutation == "unsafe_path":
        manifest["artifacts"]["wifi_mac"]["path"] = "../wifi_mac.parquet"
    elif mutation == "artifact_count":
        manifest["artifacts"]["net_nodes"]["record_count"] += 1
    elif mutation == "bool_counter":
        manifest["parser_counts"]["rejected"] = False
    else:
        raw["documents"].pop()
        completion["expected_document_counts"].pop("graph_snapshot")
        completion["document_set_sha256"] = _digest(
            sorted(raw["documents"], key=lambda doc: doc["external_id"])
        )
    if mutation not in {"missing_manifest", "receipt_digest"}:
        completion["run_manifest_sha256"] = _digest(manifest)
    with pytest.raises(ValueError):
        build_canonical_import_request_input_from_raw(
            raw, source="forgery", replace_scope_override=True
        )


def test_chat_error_counter_cannot_be_bypassed_by_clearing_failed_files():
    raw = _payload("chat_units_v4.json")
    raw["replace_scope"] = False
    raw["stats"].update(error_count=1, failed_files=0)
    raw["failures"] = []
    with pytest.raises(ValueError, match="Partial canonical"):
        build_canonical_import_request_input_from_raw(
            raw, source="partial", replace_scope_override=True
        )


def test_chat_hash_and_document_count_are_validated():
    raw = _payload("chat_units_v4.json")
    raw["documents"][0]["content"] += " changed"
    with pytest.raises(ValueError, match="content_hash"):
        validate_canonical_import_payload(raw)


def test_partial_receipt_retains_record_accounting_without_authorizing_replacement():
    raw = _payload()
    raw["replace_scope"] = False
    completion = raw["completion"]
    completion.update(complete=False, failures=["Rejected source record"])
    completion["run_manifest"]["parser_counts"].update(read=2, rejected=1)
    completion["run_manifest_sha256"] = _digest(completion["run_manifest"])
    assert (
        build_canonical_import_request_input_from_raw(raw, source="partial").replace_scope is False
    )
    with pytest.raises(ValueError, match="Partial canonical"):
        build_canonical_import_request_input_from_raw(
            raw, source="partial", replace_scope_override=True
        )
