"""Explicit documents-only snapshot publication and verified, write-free loading.

Preparation is a writer operation. It reserves a fresh directory and seals it last.
Readers consume verified bytes, never open the live database or acquire a write lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, select

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.infrastructure.concurrency.locks.write_lock import multi_store_write_lock
from local_rag_backend.infrastructure.persistence.shared.atomic_io import fsync_directory
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal
from local_rag_backend.infrastructure.persistence.sql.models import Document as SqlDocument
from local_rag_backend.infrastructure.persistence.vector.manifest import (
    expected_manifest_config_from_settings,
    manifest_path_for,
)
from local_rag_backend.settings import Settings

MAX_SNAPSHOT_FILE_BYTES = 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MANIFEST_NAME = "manifest.json"


class SnapshotError(ValueError):
    """A missing, unsealed, corrupt or incompatible snapshot was refused."""


class SnapshotFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    path: Literal["documents.sqlite3", "vectors.bin", "ids.json", "index_manifest.json"]
    bytes: int = Field(ge=0, le=MAX_SNAPSHOT_FILE_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["rag-read-snapshot"] = "rag-read-snapshot"
    schema_version: Literal[1] = 1
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_mode: Literal["sparse", "dense", "dual", "hybrid"]
    documents_count: int = Field(ge=0)
    embedding_config: dict[str, Any]
    retrieval_config: dict[str, Any]
    vector_backend: Literal["numpy", "faiss"] | None
    files: tuple[SnapshotFile, ...]


@dataclass(frozen=True)
class VerifiedSnapshot:
    manifest: SnapshotManifest
    documents: tuple[Document, ...]
    vector_bytes: bytes | None
    ids_bytes: bytes | None
    native_manifest: dict[str, Any] | None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode()


def _identity(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        _json_bytes({key: item for key, item in value.items() if key != "snapshot_id"})
    ).hexdigest()


def _safe_bytes(path: Path, *, limit: int) -> bytes:
    """Read one regular, stable inode without following a final symlink."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise SnapshotError("Snapshot member is not a bounded regular file")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(data) > limit or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise SnapshotError("Snapshot member changed during verification")
    return data


def _write_member(root: Path, name: str, data: bytes) -> SnapshotFile:
    path = root / name
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return SnapshotFile.model_validate(
        {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    )


def _fingerprint(settings: Settings) -> dict[str, Any]:
    return (
        expected_manifest_config_from_settings(settings)
        if settings.retrieval_mode != "sparse"
        else {}
    )


def _retrieval_config(settings: Settings) -> dict[str, Any]:
    """Bind settings that change retrieval independently of query arguments."""
    return {
        "hybrid_alpha": settings.hybrid_retrieval_alpha,
        "enable_reranker": settings.enable_reranker,
        "reranker_candidate_k": settings.reranker_candidate_k,
        "reranker_strategy": settings.reranker_strategy,
        "vector_backend": settings.vector_backend,
        "synthetic_dimension": settings.synthetic_embedding_dim
        if settings.synthetic_embeddings and settings.retrieval_mode != "sparse"
        else None,
    }


def prepare_snapshot(settings: Settings, destination: Path) -> SnapshotManifest:
    """Publish one SQL read transaction plus its vector generation under the common lock.

    No backup/create_all is used: only the mapped documents table is copied into a
    fresh database. An interrupted publication remains unsealed and is never read.
    Existing destinations, including empty directories and symlinks, are refused.
    """
    source_path = settings.get_database_path()
    if not source_path.is_file():
        raise SnapshotError("Snapshot preparation requires an existing SQLite document store")
    destination = destination.absolute()
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    fsync_directory(destination.parent)
    source_engine = create_engine(settings.sqlite_url)
    target_path = destination / "documents.sqlite3"
    target_engine = create_engine(f"sqlite:///{target_path}")
    files: list[SnapshotFile] = []
    count = 0
    backend: Literal["numpy", "faiss"] | None = None
    try:
        with multi_store_write_lock(
            coordination_dir=settings.get_coordination_dir(),
            timeout_s=settings.write_lock_timeout_s,
            poll_s=settings.write_lock_poll_s,
            metrics_path=settings.lock_metrics_path,
        ):
            journal = FileMutationJournal(settings.get_coordination_dir() / ".mutation_journal")
            if journal.count_incomplete():
                raise SnapshotError("Snapshot refused: unresolved mutation journal")
            with source_engine.connect() as source:
                source.exec_driver_sql("BEGIN")
                # The explicit transaction remains open throughout the vector copy.
                with target_engine.begin() as target:
                    SqlDocument.__table__.create(target)
                    rows = source.execute(
                        select(SqlDocument.__table__).order_by(SqlDocument.doc_id)
                    ).mappings()
                    for batch in rows.partitions(1000):
                        target.execute(SqlDocument.__table__.insert(), [dict(row) for row in batch])
                        count += len(batch)
                if settings.retrieval_mode in {"dense", "hybrid"}:
                    native_bytes = _safe_bytes(
                        manifest_path_for(settings.index_path), limit=MAX_MANIFEST_BYTES
                    )
                    native = json.loads(native_bytes)
                    if not isinstance(native, dict) or any(
                        native.get(key) != value for key, value in _fingerprint(settings).items()
                    ):
                        raise SnapshotError("Snapshot refused: embedding manifest drift")
                    raw_backend = native.get("index_backend")
                    if raw_backend not in {"numpy", "faiss"} or (
                        settings.vector_backend != "auto" and raw_backend != settings.vector_backend
                    ):
                        raise SnapshotError("Snapshot refused: vector backend drift")
                    backend = raw_backend
                    for name, path, limit in (
                        ("vectors.bin", Path(settings.index_path), MAX_SNAPSHOT_FILE_BYTES),
                        ("ids.json", Path(settings.id_map_path), MAX_SNAPSHOT_FILE_BYTES),
                        (
                            "index_manifest.json",
                            manifest_path_for(settings.index_path),
                            MAX_MANIFEST_BYTES,
                        ),
                    ):
                        files.append(
                            _write_member(destination, name, _safe_bytes(path, limit=limit))
                        )
                source.rollback()
        target_engine.dispose()
        documents_bytes = _safe_bytes(target_path, limit=MAX_SNAPSHOT_FILE_BYTES)
        with target_path.open("rb") as stream:
            os.fsync(stream.fileno())
        files.insert(
            0,
            SnapshotFile(
                path="documents.sqlite3",
                bytes=len(documents_bytes),
                sha256=hashlib.sha256(documents_bytes).hexdigest(),
            ),
        )
        body: dict[str, Any] = {
            "kind": "rag-read-snapshot",
            "schema_version": 1,
            "retrieval_mode": settings.retrieval_mode,
            "documents_count": count,
            "embedding_config": _fingerprint(settings),
            "retrieval_config": _retrieval_config(settings),
            "vector_backend": backend,
            "files": [item.model_dump() for item in files],
        }
        body["snapshot_id"] = _identity(body)
        manifest = SnapshotManifest.model_validate_json(_json_bytes(body))
        # Validate every copied artifact before making the seal visible.
        payloads = _verified_payloads(destination, manifest)
        documents = _documents(payloads["documents.sqlite3"], count)
        if backend is not None:
            from local_rag_backend.infrastructure.persistence.vector.read_snapshot import (
                SnapshotVectorStorage,
            )

            SnapshotVectorStorage(
                VerifiedSnapshot(
                    manifest,
                    documents,
                    payloads["vectors.bin"],
                    payloads["ids.json"],
                    json.loads(payloads["index_manifest.json"]),
                )
            )
        with (destination / ".manifest.pending").open("xb") as stream:
            stream.write(_json_bytes(body))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(destination / ".manifest.pending", destination / MANIFEST_NAME)
        fsync_directory(destination)
        return manifest
    finally:
        source_engine.dispose()
        target_engine.dispose()


def _verified_payloads(root: Path, manifest: SnapshotManifest) -> dict[str, bytes]:
    expected_names = {"documents.sqlite3"}
    if manifest.retrieval_mode in {"dense", "hybrid"}:
        expected_names.update({"vectors.bin", "ids.json", "index_manifest.json"})
    if (
        len(manifest.files) != len(expected_names)
        or {item.path for item in manifest.files} != expected_names
    ):
        raise SnapshotError("Snapshot member set is invalid")
    payloads: dict[str, bytes] = {}
    for item in manifest.files:
        data = _safe_bytes(root / item.path, limit=MAX_SNAPSHOT_FILE_BYTES)
        if len(data) != item.bytes or hashlib.sha256(data).hexdigest() != item.sha256:
            raise SnapshotError("Snapshot member digest mismatch")
        payloads[item.path] = data
    return payloads


def _documents(data: bytes, count: int) -> tuple[Document, ...]:
    # Deserializing verified bytes avoids a verify/open race, WAL or sidecar access.
    # The query-only in-memory image cannot mutate the sealed bundle or live store.
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.deserialize(data)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA trusted_schema=OFF")
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if tables != [("documents",)]:
            raise SnapshotError("Snapshot must contain the documents table only")
        rows = connection.execute(
            "SELECT doc_id, content, external_id, source_id, metadata, scope, snapshot_id "
            "FROM documents ORDER BY doc_id"
        ).fetchall()
    if len(rows) != count:
        raise SnapshotError("Snapshot document count mismatch")
    documents = []
    for row in rows:
        metadata = json.loads(row[4]) if row[4] is not None else None
        if row[5] is not None or row[6] is not None:
            metadata = dict(metadata or {})
            if row[5] is not None:
                metadata["scope"] = row[5]
            if row[6] is not None:
                metadata["snapshot_id"] = row[6]
        documents.append(
            Document(
                id=DocId(row[0]),
                content=row[1],
                external_id=row[2],
                source_id=row[3],
                metadata=metadata,
            )
        )
    return tuple(documents)


def _load_snapshot(settings: Settings) -> VerifiedSnapshot:
    """Verify a sealed bundle without model/cache/storage initialization or writes."""
    if settings.ephemeral_snapshot_dir is None:
        raise SnapshotError("No ephemeral_snapshot_dir configured; prepare a frozen snapshot")
    root = settings.ephemeral_snapshot_dir.absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise SnapshotError("Snapshot directory must be a real directory without symlink ancestors")
    raw = _safe_bytes(root / MANIFEST_NAME, limit=MAX_MANIFEST_BYTES)
    manifest = SnapshotManifest.model_validate_json(raw)
    if manifest.snapshot_id != _identity(manifest.model_dump()):
        raise SnapshotError("Snapshot seal identity mismatch")
    if (
        manifest.retrieval_mode != settings.retrieval_mode
        or manifest.embedding_config != _fingerprint(settings)
        or manifest.retrieval_config != _retrieval_config(settings)
    ):
        raise SnapshotError("Snapshot configuration drift; prepare a new generation")
    allowed = {MANIFEST_NAME, *(item.path for item in manifest.files)}
    if {path.name for path in root.iterdir()} != allowed:
        raise SnapshotError("Snapshot contains unknown members or SQLite sidecars")
    payloads = _verified_payloads(root, manifest)
    native = (
        json.loads(payloads["index_manifest.json"]) if "index_manifest.json" in payloads else None
    )
    return VerifiedSnapshot(
        manifest=manifest,
        documents=_documents(payloads["documents.sqlite3"], manifest.documents_count),
        vector_bytes=payloads.get("vectors.bin"),
        ids_bytes=payloads.get("ids.json"),
        native_manifest=native,
    )


def load_snapshot(settings: Settings) -> VerifiedSnapshot:
    """Normalize known artifact parse/read failures without exposing stored contents."""
    try:
        return _load_snapshot(settings)
    except SnapshotError:
        raise
    except (OSError, ValueError, sqlite3.DatabaseError) as exc:
        raise SnapshotError("Snapshot cannot be verified; prepare a new generation") from exc
