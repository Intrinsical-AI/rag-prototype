"""Validate explicitly identified chat and RF canonical producer envelopes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
_Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
_Count = Annotated[int, Field(ge=0, strict=True)]
_ARTIFACT_SCHEMAS = {
    "rf_meta": "rf_meta_v2",
    "wifi_mac": "wifi_mac_v2",
    "feature_windows": "feat_window_v2",
    "net_nodes": "net_node_v2",
    "net_edges": "net_edge_v2",
    "graph_snapshots": "graph_snapshot_v2",
    "graph_manifest": "graph_manifest_v1",
    "findings": "finding_v1",
    "evidence_bundles": "evidence_bundle_v1",
}
_DOC_ARTIFACTS = {
    "dataset_manifest": "graph_manifest",
    "feature_window": "feature_windows",
    "net_node": "net_nodes",
    "net_edge": "net_edges",
    "graph_snapshot": "graph_snapshots",
    "finding": "findings",
    "evidence_bundle": "evidence_bundles",
}


def _digest(value: object) -> str:
    material = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class _Envelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    scope: Annotated[str, Field(min_length=1, max_length=512)]
    snapshot_id: Annotated[str, Field(min_length=1, max_length=512)]
    replace_scope: bool
    documents: Annotated[list[dict[str, Any]], Field(min_length=1, max_length=5000)]

    @field_validator("scope", "snapshot_id")
    @classmethod
    def _identity(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("Producer identities must be nonblank and already normalized")
        return value


class _ChatProducer(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    name: Literal["chat-exports"]
    version: _Text
    format: Literal["chat-units"]
    format_version: Literal["4"]
    options: dict[str, Any]
    options_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{16}$")]


class _ChatStats(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    total_messages: _Count
    emitted_documents: Annotated[int, Field(ge=1, strict=True)]
    failed_files: _Count
    error_count: _Count


class ChatUnitsPayload(_Envelope):
    kind: Literal["chat-units"]
    schema_version: Literal["4"]
    producer: _ChatProducer
    stats: _ChatStats
    failures: list[dict[str, Any]]

    @model_validator(mode="after")
    def _coherent(self) -> ChatUnitsPayload:
        if self.stats.emitted_documents != len(self.documents):
            raise ValueError("Chat stats.emitted_documents must match documents length")
        if self.producer.options.get("format_version") != "4":
            raise ValueError("Chat producer options must identify format_version='4'")
        for document in self.documents:
            metadata = document.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError("Chat documents require metadata")
            source, key = metadata.get("source"), metadata.get("chat_key")
            if (
                source not in {"whatsapp", "telegram"}
                or not isinstance(key, str)
                or not key.strip()
            ):
                raise ValueError("Chat source and chat_key must be explicit")
            source_id = f"chat:{source}:{key}"
            if document.get("source_id") != source_id or not self.scope.startswith(source_id + ":"):
                raise ValueError("Chat documents must match their source scope")
            if metadata.get("unit_type") not in {"window", "message", "media"}:
                raise ValueError("Unknown chat unit_type")
            ids = metadata.get("message_ids")
            if (
                not isinstance(ids, list)
                or not ids
                or any(not isinstance(item, str) for item in ids)
            ):
                raise ValueError("Chat documents require message_ids")
            content = document.get("content")
            if (
                not isinstance(content, str)
                or metadata.get("content_hash")
                != hashlib.sha256(content.encode("utf-8")).hexdigest()
            ):
                raise ValueError("Chat document content_hash must match content")
        return self


class _InputIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    format: Literal["pcap", "wigle"]
    size_bytes: Annotated[int, Field(ge=1, le=256 * 1024 * 1024, strict=True)]
    sha256: _Digest


class _ParserCounts(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    read: Annotated[int, Field(ge=0, le=100_000, strict=True)]
    accepted: _Count
    out_of_domain: _Count
    rejected: _Count
    unknown: _Count

    @model_validator(mode="after")
    def _conserved(self) -> _ParserCounts:
        if self.read != self.accepted + self.out_of_domain + self.rejected + self.unknown:
            raise ValueError("Parser counters must account for every input record")
        return self


class _Artifact(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    path: _Text
    schema_version: _Text
    size_bytes: _Count
    sha256: _Digest
    record_count: _Count

    @field_validator("path")
    @classmethod
    def _relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or path.as_posix() != value
            or value == "."
        ):
            raise ValueError("Artifact receipt paths must be normalized relative paths")
        return value


class _RunManifest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["tr3v0r-run"]
    schema_version: Literal["1"]
    input: _InputIdentity
    producer_version: _Text
    config_version: _Text
    parameters_sha256: _Digest
    completed_stages: list[Literal["parse", "build", "detect"]]
    parser_counts: _ParserCounts
    artifacts: dict[str, _Artifact]

    @model_validator(mode="after")
    def _coherent(self) -> _RunManifest:
        if self.completed_stages != ["parse", "build", "detect"]:
            raise ValueError("Replacement evidence requires all completed pipeline stages")
        if set(self.artifacts) != set(_ARTIFACT_SCHEMAS):
            raise ValueError("Replacement evidence requires all pipeline artifacts")
        paths = set()
        for name, artifact in self.artifacts.items():
            if artifact.schema_version != _ARTIFACT_SCHEMAS[name] or artifact.path in paths:
                raise ValueError("Conflicting artifact schema or path")
            paths.add(artifact.path)
        if any(
            self.artifacts[name].record_count != self.parser_counts.accepted
            for name in ("rf_meta", "wifi_mac")
        ):
            raise ValueError("Raw artifact counts must match accepted parser observations")
        if self.artifacts["graph_manifest"].record_count != 1:
            raise ValueError("Replacement evidence requires exactly one graph manifest")
        return self


class _Completion(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    complete: bool
    failures: list[_Text]
    dataset_key: Annotated[str, Field(pattern=r"^[a-z0-9._-]+$")]
    expected_document_counts: dict[str, _Count]
    document_set_sha256: _Digest
    run_manifest: _RunManifest | None
    run_manifest_sha256: _Digest | None


class Tr3v0rCanonicalPayload(_Envelope):
    kind: Literal["tr3v0r-canonical-rag"]
    schema_version: Literal["2"]
    completion: _Completion

    @model_validator(mode="after")
    def _coherent(self) -> Tr3v0rCanonicalPayload:
        completion = self.completion
        if self.scope != f"tr3v0r:{completion.dataset_key}":
            raise ValueError("Tr3v0r completion dataset_key must match scope")
        counts: Counter[str] = Counter()
        for document in self.documents:
            metadata = document.get("metadata")
            if (
                not isinstance(metadata, dict)
                or metadata.get("dataset_key") != completion.dataset_key
            ):
                raise ValueError("Tr3v0r document dataset_key must match completion")
            doc_type = metadata.get("doc_type")
            if doc_type not in _DOC_ARTIFACTS:
                raise ValueError("Unknown Tr3v0r document type")
            counts[doc_type] += 1
            if not str(document.get("external_id", "")).startswith(self.scope + ":") or not str(
                document.get("source_id", "")
            ).startswith(self.scope + ":artifact:"):
                raise ValueError("Tr3v0r document identities must match scope")
        if dict(counts) != completion.expected_document_counts:
            raise ValueError("Tr3v0r expected document counts do not match documents")
        if (
            _digest(sorted(self.documents, key=lambda doc: str(doc["external_id"])))
            != completion.document_set_sha256
        ):
            raise ValueError("Tr3v0r document-set digest mismatch")
        if completion.complete:
            if completion.failures or completion.run_manifest is None:
                raise ValueError(
                    "Complete Tr3v0r exports require a complete-run manifest and no failures"
                )
            counters = completion.run_manifest.parser_counts
            if counters.accepted == 0 or counters.rejected or counters.unknown:
                raise ValueError(
                    "Complete runs require accepted observations and no rejected or unknown records"
                )
        elif not completion.failures:
            raise ValueError("Incomplete Tr3v0r exports require explicit failure evidence")
        manifest = completion.run_manifest
        if manifest is None:
            if completion.run_manifest_sha256 is not None:
                raise ValueError("Run-manifest digest requires a run manifest")
        else:
            if _digest(manifest.model_dump()) != completion.run_manifest_sha256:
                raise ValueError("Tr3v0r run-manifest digest mismatch")
            material = [
                {
                    "artifact": name,
                    "file": PurePosixPath(manifest.artifacts[name].path).name,
                    "sha256": manifest.artifacts[name].sha256,
                }
                for name in sorted(set(_ARTIFACT_SCHEMAS) - {"rf_meta", "wifi_mac"})
            ]
            if self.snapshot_id != f"{completion.dataset_key}-{_digest(material)[:16]}":
                raise ValueError("Tr3v0r snapshot_id must match artifact receipt identities")
            if any(
                counts[doc_type] != manifest.artifacts[artifact].record_count
                for doc_type, artifact in _DOC_ARTIFACTS.items()
            ):
                raise ValueError("Tr3v0r document counts must match run artifact counts")
            if completion.complete and (counts["feature_window"] == 0 or counts["net_node"] == 0):
                raise ValueError("Complete runs require represented observations")
        return self
