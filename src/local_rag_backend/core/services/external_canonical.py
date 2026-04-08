"""Validation helpers for external canonical import payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REPOGPT_CODE_UNITS_KIND = "code-units"
REPOGPT_CODE_UNITS_SCHEMA_VERSION = "4"
_REPOGPT_TOP_LEVEL_MARKERS = frozenset({"kind", "schema_version", "repo_key", "stats", "failures"})
_REPOGPT_METADATA_KEYS = ("path", "unit_type", "repo_key", "content_hash")
_REPOGPT_COHERENT_FIELDS = ("scope", "snapshot_id", "path", "unit_type", "repo_key", "content_hash")


def normalize_external_canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate known external producer payloads without specializing the core import flow."""
    normalized = dict(payload)
    if _looks_like_repogpt_code_units(normalized):
        _validate_repogpt_code_units_payload(normalized)
    return normalized


def _looks_like_repogpt_code_units(payload: Mapping[str, Any]) -> bool:
    return any(marker in payload for marker in _REPOGPT_TOP_LEVEL_MARKERS)


def _validate_repogpt_code_units_payload(payload: Mapping[str, Any]) -> None:
    kind = str(payload.get("kind") or "").strip()
    if kind != REPOGPT_CODE_UNITS_KIND:
        raise ValueError(
            f"RepoGPT canonical payloads must declare kind={REPOGPT_CODE_UNITS_KIND!r}."
        )

    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != REPOGPT_CODE_UNITS_SCHEMA_VERSION:
        raise ValueError("RepoGPT canonical payloads must use code-units schema_version='4'.")

    scope = str(payload.get("scope") or "").strip()
    snapshot_id = str(payload.get("snapshot_id") or "").strip()
    if not scope:
        raise ValueError("RepoGPT canonical payloads must include a non-blank scope.")
    if not snapshot_id:
        raise ValueError("RepoGPT canonical payloads must include a non-blank snapshot_id.")

    if "replace_scope" in payload and not isinstance(payload.get("replace_scope"), bool):
        raise ValueError("RepoGPT canonical payload replace_scope must be a boolean when provided.")

    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("RepoGPT canonical payloads must include a non-empty documents list.")

    repo_key = str(payload.get("repo_key") or "").strip()
    if not repo_key:
        raise ValueError("RepoGPT canonical payloads must include a non-blank repo_key.")

    for idx, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"RepoGPT documents[{idx}] must be an object.")
        metadata = document.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"RepoGPT documents[{idx}] must include metadata as an object.")

        external_id = str(document.get("external_id") or "").strip()
        content = str(document.get("content") or "").strip()
        if not external_id:
            raise ValueError(f"RepoGPT documents[{idx}].external_id must not be blank.")
        if not content:
            raise ValueError(f"RepoGPT documents[{idx}].content must not be blank.")

        for key in _REPOGPT_METADATA_KEYS:
            value = metadata.get(key)
            if value is None or not str(value).strip():
                raise ValueError(f"RepoGPT documents[{idx}].metadata.{key} must not be blank.")

        for key in _REPOGPT_COHERENT_FIELDS:
            doc_value = document.get(key)
            meta_value = metadata.get(key)
            top_level_value = payload.get(key)
            _assert_coherent_value(
                idx=idx,
                key=key,
                left_name="document",
                left_value=doc_value,
                right_name="metadata",
                right_value=meta_value,
            )
            if key in {"scope", "snapshot_id", "repo_key"}:
                _assert_coherent_value(
                    idx=idx,
                    key=key,
                    left_name="document",
                    left_value=doc_value,
                    right_name="payload",
                    right_value=top_level_value,
                )
                _assert_coherent_value(
                    idx=idx,
                    key=key,
                    left_name="metadata",
                    left_value=meta_value,
                    right_name="payload",
                    right_value=top_level_value,
                )


def _assert_coherent_value(
    *,
    idx: int,
    key: str,
    left_name: str,
    left_value: Any,
    right_name: str,
    right_value: Any,
) -> None:
    if left_value is None or right_value is None:
        return
    if str(left_value).strip() != str(right_value).strip():
        raise ValueError(
            f"RepoGPT documents[{idx}] {left_name}.{key} must match {right_name}.{key}."
        )
