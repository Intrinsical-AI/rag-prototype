"""Validate RepoGPT's code-units v5 wire format before canonical adaptation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPOGPT_CODE_UNITS_KIND = "code-units"
REPOGPT_CODE_UNITS_SCHEMA_VERSION = "5"
_REPOGPT_TOP_LEVEL_MARKERS = frozenset({"kind", "schema_version", "repo_key"})
_REPOGPT_UNIT_FIELDS = (
    "repo_key",
    "scope",
    "snapshot_id",
    "path",
    "language",
    "unit_type",
    "unit_level",
    "symbol",
    "qualified_name",
    "container_id",
    "depth",
    "ancestor_path",
    "start_line",
    "end_line",
    "content_hash",
    "docstring_present",
    "has_children",
    "content_ranges",
)
_NonBlank = Annotated[str, Field(min_length=1, pattern=r"\S")]
_Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
_NonNegative = Annotated[int, Field(ge=0)]


class _ProducerModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")


class _FileDigest(_ProducerModel):
    size: _NonNegative
    sha256: _Digest


class _UnitMetadata(_ProducerModel):
    file: _FileDigest
    tags: list[str]
    attributes: dict[str, Any]
    dependencies: list[Any]

    @model_validator(mode="before")
    @classmethod
    def _no_unit_field_aliases(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and set(value).intersection(_REPOGPT_UNIT_FIELDS):
            raise ValueError("RepoGPT unit fields belong at document level, not in metadata.")
        return value


class _ContentRange(_ProducerModel):
    start_line: Annotated[int, Field(ge=1)]
    end_line: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _ordered(self) -> _ContentRange:
        if self.end_line < self.start_line:
            raise ValueError("RepoGPT content_ranges end_line must not precede start_line.")
        return self


class _CodeUnit(_ProducerModel):
    external_id: _NonBlank
    source_id: _NonBlank
    repo_key: _NonBlank
    scope: _NonBlank
    snapshot_id: _NonBlank
    path: _NonBlank
    language: str | None
    unit_type: _NonBlank
    unit_level: Literal["container", "symbol"]
    symbol: str | None
    qualified_name: _NonBlank
    container_id: _NonBlank
    depth: _NonNegative
    ancestor_path: list[str]
    start_line: Annotated[int, Field(ge=1)] | None
    end_line: Annotated[int, Field(ge=1)] | None
    content: str
    content_ranges: list[_ContentRange] | None = None
    content_hash: _Digest
    docstring_present: bool
    has_children: bool
    metadata: _UnitMetadata

    @model_validator(mode="after")
    def _valid_content(self) -> _CodeUnit:
        if self.unit_type == "module":
            if self.content_ranges is None:
                raise ValueError("RepoGPT module documents require content_ranges.")
            if self.unit_level != "container" or self.container_id != self.external_id:
                raise ValueError("RepoGPT module documents must be self-contained containers.")
        elif self.content_ranges is not None:
            raise ValueError("RepoGPT content_ranges belong only on module documents.")
        elif not self.content.strip():
            raise ValueError("RepoGPT non-module documents require non-blank content.")
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.content_hash:
            raise ValueError("RepoGPT document content_hash does not match content.")
        return self


class _Stats(_ProducerModel):
    total_files: _NonNegative
    ok_files: _NonNegative
    failed_files: _NonNegative
    emitted_documents: _NonNegative


class _Failure(_ProducerModel):
    record_type: Literal["failure"]
    schema_version: Literal["5"]
    path: str
    language: str
    error: str
    file: _FileDigest


class _CodeUnitsPayload(_ProducerModel):
    schema_version: Literal["5"]
    kind: Literal["code-units"]
    repo_key: _NonBlank
    scope: _NonBlank
    snapshot_id: _NonBlank
    replace_scope: bool
    stats: _Stats
    failures: list[_Failure]
    documents: list[_CodeUnit]

    @model_validator(mode="after")
    def _coherent_documents(self) -> _CodeUnitsPayload:
        if not self.documents:
            raise ValueError("RepoGPT canonical payloads must include a non-empty documents list.")
        if self.stats.emitted_documents != len(self.documents):
            raise ValueError("RepoGPT stats.emitted_documents must match documents length.")
        for index, document in enumerate(self.documents):
            for key in ("repo_key", "scope", "snapshot_id"):
                if getattr(document, key) != getattr(self, key):
                    raise ValueError(f"RepoGPT documents[{index}].{key} must match payload.{key}.")
        return self


def normalize_external_canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep generic documents unchanged; drop validated empty module containers."""
    normalized = dict(payload)
    if not any(
        marker in payload and payload[marker] is not None for marker in _REPOGPT_TOP_LEVEL_MARKERS
    ):
        return normalized
    if payload.get("kind") != REPOGPT_CODE_UNITS_KIND:
        raise ValueError("RepoGPT canonical payloads must declare kind='code-units'.")
    if payload.get("schema_version") != REPOGPT_CODE_UNITS_SCHEMA_VERSION:
        raise ValueError("RepoGPT canonical payloads must use code-units schema_version='5'.")
    _CodeUnitsPayload.model_validate(payload)
    normalized["documents"] = [
        document
        for document in payload["documents"]
        if not (document["unit_type"] == "module" and not document["content"].strip())
    ]
    return normalized


def repogpt_document_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    """Project the validated producer unit into the canonical queryable metadata shape."""
    metadata = {
        **document["metadata"],
        **{key: document[key] for key in _REPOGPT_UNIT_FIELDS if key != "content_ranges"},
    }
    if document.get("content_ranges") is not None:
        metadata["content_ranges"] = document["content_ranges"]
    return metadata
