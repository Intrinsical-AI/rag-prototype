# src/infrastructure/persistence/sql/alchemy_engine.py
"""
SQLAlchemy-based implementation of the document and history repositories.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

from sqlalchemy import text

from local_rag_backend.core.domain.entities import Document as DomainDocument
from local_rag_backend.core.domain.types import DocId, new_doc_id
from local_rag_backend.core.ports import DocumentRepoPort, QAHistoryPort
from local_rag_backend.infrastructure.persistence.sql.base import SessionLocal
from local_rag_backend.infrastructure.persistence.sql.crud import (
    add_documents,
    add_history,
    delete_documents,
)
from local_rag_backend.infrastructure.persistence.sql.models import (
    Document as DbDocument,
    DocumentTombstone as DbDocumentTombstone,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from typing import Any, Literal

    from sqlalchemy.orm import Session, sessionmaker


def _normalize_external_ids(external_ids: Sequence[str]) -> list[str]:
    """
    Normalize external IDs for delete/tombstone operations.

    - Strip whitespace
    - Drop blanks
    - Deduplicate while preserving order
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in external_ids:
        ext = str(raw).strip()
        if not ext or ext in seen:
            continue
        seen.add(ext)
        normalized.append(ext)
    return normalized


@dataclass(frozen=True)
class _DocumentChanges:
    """Per-field change flags computed when evaluating an upsert against an existing row."""

    content: bool
    metadata: bool
    source: bool
    dedup: bool

    @property
    def any_changed(self) -> bool:
        return self.content or self.metadata or self.source or self.dedup


def _to_domain_document(d: DbDocument) -> DomainDocument:
    """Map a single ORM row to its domain entity."""
    return DomainDocument(
        id=DocId(d.doc_id),
        content=d.content,
        external_id=d.external_id,
        source_id=d.source_id,
        metadata=d.metadata_,
    )


@contextmanager
def get_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = session_factory()
    try:
        yield session
    except Exception:
        with suppress(Exception):
            session.rollback()
        raise
    finally:
        session.close()


class SystemStateStorage:
    """
    Persist process-coordination state in SQLite.

    Used by app.factory to invalidate process-local caches across workers/processes.
    """

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or SessionLocal
        self._table_ready = False
        self._table_lock = Lock()

    def _ensure_table(self, session: Session) -> None:
        if self._table_ready:
            return
        with self._table_lock:
            if self._table_ready:
                return
            session.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS system_state ("
                    "key TEXT PRIMARY KEY, "
                    "version INTEGER NOT NULL DEFAULT 0, "
                    "updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
                    ")"
                )
            )
            session.commit()
            self._table_ready = True

    def get_version(self, key: str) -> int:
        state_key = str(key).strip()
        if not state_key:
            raise ValueError("system_state key must not be blank")

        with get_session(self._session_factory) as session:
            self._ensure_table(session)
            row = session.execute(
                text("SELECT version FROM system_state WHERE key=:key"),
                {"key": state_key},
            ).first()
            if row is None:
                return 0
            return int(row[0] or 0)

    def bump_version(self, key: str) -> int:
        state_key = str(key).strip()
        if not state_key:
            raise ValueError("system_state key must not be blank")

        with get_session(self._session_factory) as session:
            self._ensure_table(session)
            session.execute(
                text(
                    "INSERT INTO system_state(key, version, updated_at) "
                    "VALUES(:key, 0, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(key) DO NOTHING"
                ),
                {"key": state_key},
            )
            session.execute(
                text(
                    "UPDATE system_state "
                    "SET version = version + 1, updated_at = CURRENT_TIMESTAMP "
                    "WHERE key = :key"
                ),
                {"key": state_key},
            )
            version = session.execute(
                text("SELECT version FROM system_state WHERE key=:key"),
                {"key": state_key},
            ).scalar()
            session.commit()
            return int(version or 0)


class SqlDocumentStorage(DocumentRepoPort):
    """SQL-based implementation of the document repository port."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._session_factory = session_factory or SessionLocal

    def store_documents(self, texts: Sequence[str]) -> list[DocId]:
        """Store documents in the database."""
        with get_session(self._session_factory) as session:
            return add_documents(session, list(texts))

    def delete_documents(self, ids: Sequence[DocId]) -> None:
        """Delete documents by IDs (best-effort rollback helper for ETL)."""
        with get_session(self._session_factory) as session:
            delete_documents(session, list(ids))

    def get(self, ids: Sequence[DocId]) -> Sequence[DomainDocument]:
        """Retrieve documents by their IDs."""
        normalized = [str(x) for x in ids if str(x).strip()]
        if not normalized:
            return []
        with get_session(self._session_factory) as session:
            db_docs = session.query(DbDocument).filter(DbDocument.doc_id.in_(normalized)).all()
            return [_to_domain_document(d) for d in db_docs]

    def get_all_documents(self) -> Sequence[DomainDocument]:
        """Retrieve all documents from the database."""
        with get_session(self._session_factory) as session:
            db_docs = session.query(DbDocument).order_by(DbDocument.doc_id).all()
            return [_to_domain_document(d) for d in db_docs]

    @dataclass(frozen=True)
    class UpsertDoc:
        external_id: str
        content: str
        source_id: str | None = None
        metadata: Mapping[str, Any] | None = None
        chunk_dedup_sha256: str | None = None

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

    def get_existing_doc_states_by_external_id(
        self, external_ids: Sequence[str]
    ) -> dict[str, ExistingDocState]:
        ext_ids = [str(x).strip() for x in external_ids if str(x).strip()]
        if not ext_ids:
            return {}
        with get_session(self._session_factory) as session:
            rows = (
                session.query(
                    DbDocument.doc_id,
                    DbDocument.external_id,
                    DbDocument.content,
                    DbDocument.content_sha256,
                )
                .filter(DbDocument.external_id.is_not(None))
                .filter(DbDocument.external_id.in_(ext_ids))
                .all()
            )
            out: dict[str, SqlDocumentStorage.ExistingDocState] = {}
            for doc_id, ext_id, content, content_sha in rows:
                if ext_id is None:
                    continue
                out[str(ext_id)] = SqlDocumentStorage.ExistingDocState(
                    id=DocId(str(doc_id)),
                    external_id=str(ext_id),
                    content=str(content or ""),
                    content_sha256=(str(content_sha) if content_sha is not None else None),
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

        with get_session(self._session_factory) as session:
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

            session.commit()

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

        with get_session(self._session_factory) as session:
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

    def get_tombstoned_external_ids(self, external_ids: Sequence[str]) -> set[str]:
        ext_ids = _normalize_external_ids(external_ids)
        if not ext_ids:
            return set()
        with get_session(self._session_factory) as session:
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

        with get_session(self._session_factory) as session:
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
            session.commit()
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

        with get_session(self._session_factory) as session:
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
            session.commit()
            return int(deleted_sql or 0), deleted_ids, missing, len(to_tombstone)


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

    dedup_changed = False
    if item.chunk_dedup_sha256 is not None:
        dedup_changed = item.chunk_dedup_sha256 != db_doc.chunk_dedup_sha256

    return _DocumentChanges(
        content=content_changed,
        metadata=metadata_changed,
        source=source_changed,
        dedup=dedup_changed,
    )


class HistorySqlStorage(QAHistoryPort):
    """SQL-based implementation of the history repository port."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._session_factory = session_factory or SessionLocal

    def save(self, q: str, a: str, source_ids: Sequence[DocId]) -> None:
        """Save a question-answer pair to the history table."""
        with get_session(self._session_factory) as session:
            add_history(session, q, a, source_ids=list(source_ids))
