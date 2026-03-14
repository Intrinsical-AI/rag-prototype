"""Document repository adapter backed by SQLAlchemy."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from local_rag_backend.core.domain.entities import Document as DomainDocument
from local_rag_backend.core.domain.types import DocId, new_doc_id
from local_rag_backend.core.ports import DocumentRepoPort
from local_rag_backend.infrastructure.persistence.sql import base as db_base
from local_rag_backend.infrastructure.persistence.sql.crud import add_documents, delete_documents
from local_rag_backend.infrastructure.persistence.sql.models import (
    Document as DbDocument,
    DocumentTombstone as DbDocumentTombstone,
)
from local_rag_backend.infrastructure.persistence.sql.sessions import (
    _normalize_external_ids,
    get_managed_session,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from sqlalchemy.orm import Session, sessionmaker


@dataclass(frozen=True)
class _DocumentChanges:
    """Per-field change flags computed when evaluating an upsert against an existing row."""

    content: bool
    metadata: bool
    source: bool
    scope: bool
    snapshot: bool
    dedup: bool

    @property
    def any_changed(self) -> bool:
        return self.content or self.metadata or self.source or self.scope or self.snapshot or self.dedup


def _to_domain_document(d: DbDocument) -> DomainDocument:
    """Map a single ORM row to its domain entity."""
    metadata: dict[str, Any] | None = dict(d.metadata_ or {}) if d.metadata_ is not None else None
    if d.scope is not None or d.snapshot_id is not None:
        metadata = metadata or {}
        if d.scope is not None:
            metadata["scope"] = d.scope
        if d.snapshot_id is not None:
            metadata["snapshot_id"] = d.snapshot_id
    return DomainDocument(
        id=DocId(d.doc_id),
        content=d.content,
        external_id=d.external_id,
        source_id=d.source_id,
        metadata=metadata,
    )


class SqlDocumentStorage(DocumentRepoPort):
    """SQL-based implementation of the document repository port."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._session_factory = session_factory or db_base.SessionLocal

    def store_documents(self, texts: Sequence[str]) -> list[DocId]:
        """Store documents in the database."""
        with get_managed_session(self._session_factory) as (session, owns_session):
            return add_documents(session, list(texts), autocommit=owns_session)

    def delete_documents(self, ids: Sequence[DocId]) -> None:
        """Delete documents by IDs (best-effort rollback helper for ETL)."""
        with get_managed_session(self._session_factory) as (session, owns_session):
            delete_documents(session, list(ids), autocommit=owns_session)

    def get(self, ids: Sequence[DocId]) -> Sequence[DomainDocument]:
        """Retrieve documents by their IDs."""
        normalized = [str(x) for x in ids if str(x).strip()]
        if not normalized:
            return []
        with get_managed_session(self._session_factory) as (session, _owns_session):
            db_docs = session.query(DbDocument).filter(DbDocument.doc_id.in_(normalized)).all()
            docs_by_id = {str(d.doc_id): _to_domain_document(d) for d in db_docs}
            # Preserve caller order deterministically across SQLite/Python versions.
            return [docs_by_id[doc_id] for doc_id in normalized if doc_id in docs_by_id]

    def get_all_documents(self) -> Sequence[DomainDocument]:
        """Retrieve all documents from the database."""
        with get_managed_session(self._session_factory) as (session, _owns_session):
            db_docs = session.query(DbDocument).order_by(DbDocument.doc_id).all()
            return [_to_domain_document(d) for d in db_docs]

    @dataclass(frozen=True)
    class UpsertDoc:
        external_id: str
        content: str
        source_id: str | None = None
        scope: str | None = None
        snapshot_id: str | None = None
        metadata: Mapping[str, Any] | None = None
        chunk_dedup_sha256: str | None = None
        embedding: Sequence[float] | None = None

    @dataclass(frozen=True)
    class UpsertResult:
        external_id: str
        id: DocId
        action: Literal["inserted", "updated", "unchanged"]
        content_changed: bool

    @dataclass(frozen=True)
    class ExistingDocState:
        id: DocId
        external_id: str
        content: str
        content_sha256: str | None
        scope: str | None
        snapshot_id: str | None

    @dataclass(frozen=True)
    class DocumentSnapshot:
        id: DocId
        external_id: str | None
        content: str
        source_id: str | None
        scope: str | None
        snapshot_id: str | None
        metadata: Mapping[str, Any] | None
        content_sha256: str | None
        chunk_dedup_sha256: str | None

        def to_dict(self) -> dict[str, Any]:
            return {
                "id": str(self.id),
                "external_id": self.external_id,
                "content": self.content,
                "source_id": self.source_id,
                "scope": self.scope,
                "snapshot_id": self.snapshot_id,
                "metadata": dict(self.metadata) if self.metadata is not None else None,
                "content_sha256": self.content_sha256,
                "chunk_dedup_sha256": self.chunk_dedup_sha256,
            }

    def get_existing_doc_states_by_external_id(
        self, external_ids: Sequence[str]
    ) -> dict[str, ExistingDocState]:
        ext_ids = [str(x).strip() for x in external_ids if str(x).strip()]
        if not ext_ids:
            return {}
        with get_managed_session(self._session_factory) as (session, _owns_session):
            rows = (
                session.query(
                    DbDocument.doc_id,
                    DbDocument.external_id,
                    DbDocument.content,
                    DbDocument.content_sha256,
                    DbDocument.scope,
                    DbDocument.snapshot_id,
                )
                .filter(DbDocument.external_id.is_not(None))
                .filter(DbDocument.external_id.in_(ext_ids))
                .all()
            )
            out: dict[str, SqlDocumentStorage.ExistingDocState] = {}
            for doc_id, ext_id, content, content_sha, scope, snapshot_id in rows:
                if ext_id is None:
                    continue
                out[str(ext_id)] = SqlDocumentStorage.ExistingDocState(
                    id=DocId(str(doc_id)),
                    external_id=str(ext_id),
                    content=str(content or ""),
                    content_sha256=(str(content_sha) if content_sha is not None else None),
                    scope=(str(scope) if scope is not None else None),
                    snapshot_id=(str(snapshot_id) if snapshot_id is not None else None),
                )
            return out

    def upsert_documents_by_external_id(
        self, items: Sequence[UpsertDoc]
    ) -> tuple[list[UpsertResult], list[tuple[DocId, str]], list[DocId]]:
        """
        Upsert documents by `external_id` (idempotent).

        Returns:
            (results, changed_content, updated_content_ids)

        Where:
            - results: per-item action summary
            - changed_content: list[(doc_id, new_content)] for inserts + content updates
            - updated_content_ids: list[doc_id] that existed and had content changed (for index delete)
        """
        items_list = list(items)
        if not items_list:
            return [], [], []

        ext_ids = [i.external_id.strip() for i in items_list]
        if any(not x for x in ext_ids):
            raise ValueError("external_id must not be blank")
        if len(set(ext_ids)) != len(ext_ids):
            raise ValueError("external_id values must be unique within the request")

        results: list[SqlDocumentStorage.UpsertResult] = []
        changed_content: list[tuple[DocId, str]] = []
        updated_content_ids: list[DocId] = []

        with get_managed_session(self._session_factory) as (session, owns_session):
            existing = session.query(DbDocument).filter(DbDocument.external_id.in_(ext_ids)).all()
            by_external_id = {d.external_id: d for d in existing if d.external_id is not None}

            for item in items_list:
                external_id = item.external_id.strip()
                content = item.content.strip()
                if not content:
                    raise ValueError("content must not be blank")
                sha = hashlib.sha256(content.encode("utf-8")).hexdigest()

                db_doc = by_external_id.get(external_id)
                if db_doc is None:
                    new_id = str(new_doc_id())
                    new_doc = DbDocument(
                        doc_id=new_id,
                        content=content,
                        external_id=external_id,
                        source_id=item.source_id,
                        scope=item.scope,
                        snapshot_id=item.snapshot_id,
                        metadata_=dict(item.metadata) if item.metadata is not None else None,
                        content_sha256=sha,
                        chunk_dedup_sha256=item.chunk_dedup_sha256,
                    )
                    session.add(new_doc)
                    results.append(
                        SqlDocumentStorage.UpsertResult(
                            external_id=external_id,
                            id=DocId(new_id),
                            action="inserted",
                            content_changed=True,
                        )
                    )
                    changed_content.append((DocId(new_id), content))
                    continue

                changes = _detect_document_changes(db_doc, item, content, sha)

                if not changes.any_changed:
                    results.append(
                        SqlDocumentStorage.UpsertResult(
                            external_id=external_id,
                            id=DocId(db_doc.doc_id),
                            action="unchanged",
                            content_changed=False,
                        )
                    )
                    continue

                if changes.content:
                    db_doc.content = content
                    db_doc.content_sha256 = sha
                    updated_content_ids.append(DocId(db_doc.doc_id))
                    changed_content.append((DocId(db_doc.doc_id), content))

                if item.source_id is not None:
                    db_doc.source_id = item.source_id
                if item.scope is not None:
                    db_doc.scope = item.scope
                if item.snapshot_id is not None:
                    db_doc.snapshot_id = item.snapshot_id
                if item.metadata is not None:
                    db_doc.metadata_ = dict(item.metadata)
                if item.chunk_dedup_sha256 is not None:
                    db_doc.chunk_dedup_sha256 = item.chunk_dedup_sha256

                results.append(
                    SqlDocumentStorage.UpsertResult(
                        external_id=external_id,
                        id=DocId(db_doc.doc_id),
                        action="updated",
                        content_changed=changes.content,
                    )
                )

            if owns_session:
                session.commit()
            else:
                session.flush()

        return results, changed_content, updated_content_ids

    def list_ids_by_external_id_prefix(self, prefix: str) -> list[tuple[DocId, str]]:
        """
        Return existing (id, external_id) for rows whose external_id starts with `prefix`.

        Used by file ingestion to delete stale chunks when a source shrinks or its chunking changes.
        Callers should include a delimiter in the prefix to avoid accidental collisions.
        """
        prefix_s = str(prefix)
        if not prefix_s:
            return []

        escaped = prefix_s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = escaped + "%"

        with get_managed_session(self._session_factory) as (session, _owns_session):
            rows = (
                session.query(DbDocument.doc_id, DbDocument.external_id)
                .filter(DbDocument.external_id.is_not(None))
                .filter(DbDocument.external_id.like(pattern, escape="\\"))
                .all()
            )
            out: list[tuple[DocId, str]] = []
            for doc_id, ext_id in rows:
                if ext_id is None:
                    continue
                out.append((DocId(str(doc_id)), str(ext_id)))
            return out

    def snapshot_by_ids(self, ids: Sequence[DocId]) -> list[dict[str, Any]]:
        normalized = [str(x).strip() for x in ids if str(x).strip()]
        if not normalized:
            return []
        with get_managed_session(self._session_factory) as (session, _owns_session):
            rows = (
                session.query(
                    DbDocument.doc_id,
                    DbDocument.external_id,
                    DbDocument.content,
                    DbDocument.source_id,
                    DbDocument.scope,
                    DbDocument.snapshot_id,
                    DbDocument.metadata_,
                    DbDocument.content_sha256,
                    DbDocument.chunk_dedup_sha256,
                )
                .filter(DbDocument.doc_id.in_(normalized))
                .all()
            )
            snapshots = [
                SqlDocumentStorage.DocumentSnapshot(
                    id=DocId(str(doc_id)),
                    external_id=(str(external_id) if external_id is not None else None),
                    content=str(content),
                    source_id=(str(source_id) if source_id is not None else None),
                    scope=(str(scope) if scope is not None else None),
                    snapshot_id=(str(snapshot_id) if snapshot_id is not None else None),
                    metadata=(dict(metadata_) if isinstance(metadata_, dict) else None),
                    content_sha256=(str(content_sha256) if content_sha256 is not None else None),
                    chunk_dedup_sha256=(
                        str(chunk_dedup_sha256) if chunk_dedup_sha256 is not None else None
                    ),
                )
                for (
                    doc_id,
                    external_id,
                    content,
                    source_id,
                    scope,
                    snapshot_id,
                    metadata_,
                    content_sha256,
                    chunk_dedup_sha256,
                ) in rows
            ]
            return [snap.to_dict() for snap in snapshots]

    def snapshot_by_external_ids(self, external_ids: Sequence[str]) -> list[dict[str, Any]]:
        ext_ids = _normalize_external_ids(external_ids)
        if not ext_ids:
            return []
        with get_managed_session(self._session_factory) as (session, _owns_session):
            rows = (
                session.query(
                    DbDocument.doc_id,
                    DbDocument.external_id,
                    DbDocument.content,
                    DbDocument.source_id,
                    DbDocument.scope,
                    DbDocument.snapshot_id,
                    DbDocument.metadata_,
                    DbDocument.content_sha256,
                    DbDocument.chunk_dedup_sha256,
                )
                .filter(DbDocument.external_id.is_not(None))
                .filter(DbDocument.external_id.in_(ext_ids))
                .all()
            )
            snapshots = [
                SqlDocumentStorage.DocumentSnapshot(
                    id=DocId(str(doc_id)),
                    external_id=(str(external_id) if external_id is not None else None),
                    content=str(content),
                    source_id=(str(source_id) if source_id is not None else None),
                    scope=(str(scope) if scope is not None else None),
                    snapshot_id=(str(snapshot_id) if snapshot_id is not None else None),
                    metadata=(dict(metadata_) if isinstance(metadata_, dict) else None),
                    content_sha256=(str(content_sha256) if content_sha256 is not None else None),
                    chunk_dedup_sha256=(
                        str(chunk_dedup_sha256) if chunk_dedup_sha256 is not None else None
                    ),
                )
                for (
                    doc_id,
                    external_id,
                    content,
                    source_id,
                    scope,
                    snapshot_id,
                    metadata_,
                    content_sha256,
                    chunk_dedup_sha256,
                ) in rows
            ]
            return [snap.to_dict() for snap in snapshots]

    def hard_delete_by_external_ids(self, external_ids: Sequence[str]) -> int:
        ext_ids = _normalize_external_ids(external_ids)
        if not ext_ids:
            return 0
        with get_managed_session(self._session_factory) as (session, owns_session):
            deleted = (
                session.query(DbDocument)
                .filter(DbDocument.external_id.is_not(None))
                .filter(DbDocument.external_id.in_(ext_ids))
                .delete(synchronize_session=False)
            )
            if owns_session:
                session.commit()
            else:
                session.flush()
            return int(deleted or 0)

    def delete_tombstones(self, external_ids: Sequence[str]) -> int:
        ext_ids = _normalize_external_ids(external_ids)
        if not ext_ids:
            return 0
        with get_managed_session(self._session_factory) as (session, owns_session):
            deleted = (
                session.query(DbDocumentTombstone)
                .filter(DbDocumentTombstone.external_id.in_(ext_ids))
                .delete(synchronize_session=False)
            )
            if owns_session:
                session.commit()
            else:
                session.flush()
            return int(deleted or 0)

    def restore_from_snapshots(self, snapshots: Sequence[Mapping[str, Any]]) -> int:
        snapshots_list = [dict(s) for s in snapshots if isinstance(s, Mapping)]
        if not snapshots_list:
            return 0
        ids = [
            str(s.get("id") or "").strip() for s in snapshots_list if str(s.get("id") or "").strip()
        ]
        if not ids:
            return 0

        with get_managed_session(self._session_factory) as (session, owns_session):
            existing_docs = session.query(DbDocument).filter(DbDocument.doc_id.in_(ids)).all()
            by_id = {str(doc.doc_id): doc for doc in existing_docs}
            restored = 0
            for snap in snapshots_list:
                doc_id = str(snap.get("id") or "").strip()
                if not doc_id:
                    continue
                content = str(snap.get("content") or "").strip()
                if not content:
                    continue
                content_sha = (
                    str(snap.get("content_sha256") or "").strip()
                    or hashlib.sha256(content.encode("utf-8")).hexdigest()
                )
                external_id_raw = snap.get("external_id")
                external_id = str(external_id_raw) if external_id_raw is not None else None
                source_id_raw = snap.get("source_id")
                source_id = str(source_id_raw) if source_id_raw is not None else None
                scope_raw = snap.get("scope")
                scope = str(scope_raw) if scope_raw is not None else None
                snapshot_id_raw = snap.get("snapshot_id")
                snapshot_id = str(snapshot_id_raw) if snapshot_id_raw is not None else None
                metadata_raw = snap.get("metadata")
                metadata_ = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else None
                chunk_dedup_raw = snap.get("chunk_dedup_sha256")
                chunk_dedup = str(chunk_dedup_raw) if chunk_dedup_raw is not None else None

                row = by_id.get(doc_id)
                if row is None:
                    session.add(
                        DbDocument(
                            doc_id=doc_id,
                            content=content,
                            external_id=external_id,
                            source_id=source_id,
                            scope=scope,
                            snapshot_id=snapshot_id,
                            metadata_=metadata_,
                            content_sha256=content_sha,
                            chunk_dedup_sha256=chunk_dedup,
                        )
                    )
                else:
                    row.content = content
                    row.external_id = external_id
                    row.source_id = source_id
                    row.scope = scope
                    row.snapshot_id = snapshot_id
                    row.metadata_ = metadata_
                    row.content_sha256 = content_sha
                    row.chunk_dedup_sha256 = chunk_dedup
                restored += 1
            if owns_session:
                session.commit()
            else:
                session.flush()
            return restored

    def get_tombstoned_external_ids(self, external_ids: Sequence[str]) -> set[str]:
        ext_ids = _normalize_external_ids(external_ids)
        if not ext_ids:
            return set()
        with get_managed_session(self._session_factory) as (session, _owns_session):
            rows = (
                session.query(DbDocumentTombstone.external_id)
                .filter(DbDocumentTombstone.external_id.in_(ext_ids))
                .all()
            )
            return {str(r[0]) for r in rows if r and r[0]}

    def tombstone_external_ids(self, external_ids: Sequence[str]) -> int:
        ext_ids = _normalize_external_ids(external_ids)
        if not ext_ids:
            return 0

        with get_managed_session(self._session_factory) as (session, owns_session):
            existing = (
                session.query(DbDocumentTombstone.external_id)
                .filter(DbDocumentTombstone.external_id.in_(ext_ids))
                .all()
            )
            existing_set = {str(r[0]) for r in existing if r and r[0]}
            to_add = [e for e in ext_ids if e not in existing_set]
            if not to_add:
                return 0
            session.add_all([DbDocumentTombstone(external_id=e) for e in to_add])
            if owns_session:
                session.commit()
            else:
                session.flush()
            return len(to_add)

    def delete_by_external_ids(
        self, external_ids: Sequence[str]
    ) -> tuple[int, list[DocId], list[str], int]:
        """
        Hard-delete documents by external_id and add tombstones.

        Returns: (deleted_sql, deleted_ids, missing_external_ids, tombstoned)
        """
        ext_ids = _normalize_external_ids(external_ids)
        if not ext_ids:
            return 0, [], [], 0

        with get_managed_session(self._session_factory) as (session, owns_session):
            rows = (
                session.query(DbDocument.doc_id, DbDocument.external_id)
                .filter(DbDocument.external_id.is_not(None))
                .filter(DbDocument.external_id.in_(ext_ids))
                .all()
            )
            found_by_ext = {str(ext): DocId(str(doc_id)) for doc_id, ext in rows if ext is not None}
            missing = [e for e in ext_ids if e not in found_by_ext]

            existing_ts = (
                session.query(DbDocumentTombstone.external_id)
                .filter(DbDocumentTombstone.external_id.in_(ext_ids))
                .all()
            )
            existing_ts_set = {str(r[0]) for r in existing_ts if r and r[0]}
            to_tombstone = [e for e in ext_ids if e not in existing_ts_set]
            if to_tombstone:
                session.add_all([DbDocumentTombstone(external_id=e) for e in to_tombstone])

            deleted_ids = list(found_by_ext.values())
            deleted_sql = 0
            if deleted_ids:
                deleted_sql = (
                    session.query(DbDocument)
                    .filter(DbDocument.doc_id.in_([str(x) for x in deleted_ids]))
                    .delete(synchronize_session=False)
                )
            if owns_session:
                session.commit()
            else:
                session.flush()
            return int(deleted_sql or 0), deleted_ids, missing, len(to_tombstone)

    def list_external_ids_by_scope(self, scope: str) -> list[str]:
        scope_s = str(scope).strip()
        if not scope_s:
            return []
        with get_managed_session(self._session_factory) as (session, _owns_session):
            rows = (
                session.query(DbDocument.external_id)
                .filter(DbDocument.scope == scope_s)
                .filter(DbDocument.external_id.is_not(None))
                .order_by(DbDocument.external_id.asc())
                .all()
            )
            return [str(row[0]) for row in rows if row and row[0] is not None]


def _detect_document_changes(
    db_doc: DbDocument,
    item: SqlDocumentStorage.UpsertDoc,
    new_content: str,
    new_sha: str,
) -> _DocumentChanges:
    """Compare an incoming UpsertDoc against the persisted row to find what changed."""
    content_changed = (db_doc.content_sha256 or "") != new_sha or db_doc.content != new_content

    metadata_changed = False
    if item.metadata is not None:
        metadata_changed = dict(item.metadata) != (db_doc.metadata_ or {})

    source_changed = False
    if item.source_id is not None:
        source_changed = item.source_id != db_doc.source_id

    scope_changed = False
    if item.scope is not None:
        scope_changed = item.scope != db_doc.scope

    snapshot_changed = False
    if item.snapshot_id is not None:
        snapshot_changed = item.snapshot_id != db_doc.snapshot_id

    dedup_changed = False
    if item.chunk_dedup_sha256 is not None:
        dedup_changed = item.chunk_dedup_sha256 != db_doc.chunk_dedup_sha256

    return _DocumentChanges(
        content=content_changed,
        metadata=metadata_changed,
        source=source_changed,
        scope=scope_changed,
        snapshot=snapshot_changed,
        dedup=dedup_changed,
    )


__all__ = ["SqlDocumentStorage"]
