# src/infrastructure/persistence/sqlalchemy/sql_.py
"""
SQLAlchemy-based implementation of the document and history repositories.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from local_rag_backend.core.domain.entities import Document as DomainDocument
from local_rag_backend.core.ports import DocumentRepoPort, QAHistoryPort
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import SessionLocal
from local_rag_backend.infrastructure.persistence.sqlalchemy.crud import (
    add_documents,
    add_history,
    delete_documents,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy.models import Document as DbDocument
from local_rag_backend.infrastructure.persistence.sqlalchemy.models import (
    DocumentTombstone as DbDocumentTombstone,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from typing import Any, Literal

    from sqlalchemy.orm import Session, sessionmaker


@contextmanager
def get_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = session_factory()
    try:
        yield session
    except Exception:
        # Even though most CRUD helpers commit explicitly, ensure any partially-open
        # transaction is rolled back so connections don't keep locks.
        with suppress(Exception):
            session.rollback()
        raise
    finally:
        session.close()


class SqlDocumentStorage(DocumentRepoPort):
    """SQL-based implementation of the document repository port."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._session_factory = session_factory or SessionLocal

    def store_documents(self, texts: Sequence[str]) -> list[int]:
        """Store documents in the database."""
        with get_session(self._session_factory) as session:
            return add_documents(session, list(texts))

    def delete_documents(self, ids: Sequence[int]) -> None:
        """Delete documents by IDs (best-effort rollback helper for ETL)."""
        with get_session(self._session_factory) as session:
            delete_documents(session, list(ids))

    def get(self, ids: Sequence[int]) -> Sequence[DomainDocument]:
        """Retrieve documents by their IDs."""
        with get_session(self._session_factory) as session:
            db_docs = session.query(DbDocument).filter(DbDocument.id.in_(ids)).all()
            return [
                DomainDocument(
                    id=d.id,
                    content=d.content,
                    external_id=getattr(d, "external_id", None),
                    source_id=getattr(d, "source_id", None),
                    metadata=getattr(d, "metadata_", None),
                    content_sha256=getattr(d, "content_sha256", None),
                    created_at=getattr(d, "created_at", None),
                    updated_at=getattr(d, "updated_at", None),
                )
                for d in db_docs
            ]

    def get_all_documents(self) -> Sequence[DomainDocument]:
        """Retrieve all documents from the database."""
        with get_session(self._session_factory) as session:
            db_docs = session.query(DbDocument).order_by(DbDocument.id).all()
            return [
                DomainDocument(
                    id=d.id,
                    content=d.content,
                    external_id=getattr(d, "external_id", None),
                    source_id=getattr(d, "source_id", None),
                    metadata=getattr(d, "metadata_", None),
                    content_sha256=getattr(d, "content_sha256", None),
                    created_at=getattr(d, "created_at", None),
                    updated_at=getattr(d, "updated_at", None),
                )
                for d in db_docs
            ]

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
        id: int
        action: Literal["inserted", "updated", "unchanged"]
        content_changed: bool

    @dataclass(frozen=True)
    class ExistingDocState:
        id: int
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
                    DbDocument.id,
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
                    id=int(doc_id),
                    external_id=str(ext_id),
                    content=str(content or ""),
                    content_sha256=(str(content_sha) if content_sha is not None else None),
                )
            return out

    def upsert_documents_by_external_id(
        self, items: Sequence[UpsertDoc]
    ) -> tuple[list[UpsertResult], list[tuple[int, str]], list[int]]:
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
        changed_content: list[tuple[int, str]] = []
        updated_content_ids: list[int] = []

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
                    new_doc = DbDocument(
                        content=content,
                        external_id=external_id,
                        source_id=item.source_id,
                        metadata_=dict(item.metadata) if item.metadata is not None else None,
                        content_sha256=sha,
                        chunk_dedup_sha256=item.chunk_dedup_sha256,
                    )
                    session.add(new_doc)
                    session.flush()  # allocate PK
                    assert new_doc.id is not None
                    results.append(
                        SqlDocumentStorage.UpsertResult(
                            external_id=external_id,
                            id=int(new_doc.id),
                            action="inserted",
                            content_changed=True,
                        )
                    )
                    changed_content.append((int(new_doc.id), content))
                    continue

                old_sha = getattr(db_doc, "content_sha256", None) or ""
                content_changed = (old_sha != sha) or (getattr(db_doc, "content", "") != content)

                metadata_changed = False
                if item.metadata is not None:
                    current_md = getattr(db_doc, "metadata_", None)
                    metadata_changed = dict(item.metadata) != (current_md or {})

                source_changed = False
                if item.source_id is not None:
                    source_changed = item.source_id != getattr(db_doc, "source_id", None)

                dedup_changed = False
                if item.chunk_dedup_sha256 is not None:
                    dedup_changed = item.chunk_dedup_sha256 != getattr(
                        db_doc, "chunk_dedup_sha256", None
                    )

                if not (content_changed or metadata_changed or source_changed or dedup_changed):
                    results.append(
                        SqlDocumentStorage.UpsertResult(
                            external_id=external_id,
                            id=int(db_doc.id),
                            action="unchanged",
                            content_changed=False,
                        )
                    )
                    continue

                if content_changed:
                    db_doc.content = content
                    db_doc.content_sha256 = sha
                    updated_content_ids.append(int(db_doc.id))
                    changed_content.append((int(db_doc.id), content))

                if item.source_id is not None:
                    db_doc.source_id = item.source_id
                if item.metadata is not None:
                    db_doc.metadata_ = dict(item.metadata)
                if item.chunk_dedup_sha256 is not None:
                    db_doc.chunk_dedup_sha256 = item.chunk_dedup_sha256

                results.append(
                    SqlDocumentStorage.UpsertResult(
                        external_id=external_id,
                        id=int(db_doc.id),
                        action="updated",
                        content_changed=content_changed,
                    )
                )

            session.commit()

        return results, changed_content, updated_content_ids

    def list_ids_by_external_id_prefix(self, prefix: str) -> list[tuple[int, str]]:
        """
        Return existing (id, external_id) for rows whose external_id starts with `prefix`.

        Used by file ingestion to delete stale chunks when a source shrinks or its chunking changes.
        Callers should include a delimiter in the prefix to avoid accidental collisions
        (e.g. `file:/tmp/foo:` should not match `file:/tmp/foo2:`).
        """
        prefix_s = str(prefix)
        if not prefix_s:
            return []

        # Escape LIKE wildcards so prefixes containing '%' or '_' behave as literals.
        escaped = prefix_s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = escaped + "%"

        with get_session(self._session_factory) as session:
            rows = (
                session.query(DbDocument.id, DbDocument.external_id)
                .filter(DbDocument.external_id.is_not(None))
                .filter(DbDocument.external_id.like(pattern, escape="\\"))
                .all()
            )
            out: list[tuple[int, str]] = []
            for doc_id, ext_id in rows:
                if ext_id is None:
                    continue
                out.append((int(doc_id), str(ext_id)))
            return out

    def get_tombstoned_external_ids(self, external_ids: Sequence[str]) -> set[str]:
        ext_ids = [str(x).strip() for x in external_ids if str(x).strip()]
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
        ext_ids = [str(x).strip() for x in external_ids if str(x).strip()]
        if not ext_ids:
            return 0

        with get_session(self._session_factory) as session:
            # SQLite doesn't support INSERT ... ON CONFLICT in ORM portably without Core.
            # We'll ignore duplicates by checking existing first.
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
    ) -> tuple[int, list[int], list[str], int]:
        """
        Hard-delete documents by external_id and add tombstones.

        Returns: (deleted_sql, deleted_ids, missing_external_ids, tombstoned)
        """
        ext_ids = [str(x).strip() for x in external_ids if str(x).strip()]
        if not ext_ids:
            return 0, [], [], 0

        with get_session(self._session_factory) as session:
            rows = (
                session.query(DbDocument.id, DbDocument.external_id)
                .filter(DbDocument.external_id.is_not(None))
                .filter(DbDocument.external_id.in_(ext_ids))
                .all()
            )
            found_by_ext = {str(ext): int(doc_id) for doc_id, ext in rows if ext is not None}
            missing = [e for e in ext_ids if e not in found_by_ext]

            # Tombstone all requested external_ids (including missing) to prevent reappearance.
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
                    .filter(DbDocument.id.in_(deleted_ids))
                    .delete(synchronize_session=False)
                )
            session.commit()
            return int(deleted_sql or 0), deleted_ids, missing, len(to_tombstone)


class HistorySqlStorage(QAHistoryPort):
    """SQL-based implementation of the history repository port."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._session_factory = session_factory or SessionLocal

    def save(self, q: str, a: str, source_ids: Sequence[int]) -> None:
        """Save a question-answer pair to the history table."""
        with get_session(self._session_factory) as session:
            add_history(session, q, a, source_ids=list(source_ids))
