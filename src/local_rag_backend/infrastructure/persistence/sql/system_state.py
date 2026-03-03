"""System-state repository for cross-process runtime cache invalidation."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

from sqlalchemy import text

from local_rag_backend.infrastructure.persistence.sql import base as db_base
from local_rag_backend.infrastructure.persistence.sql.sessions import get_session

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker


class SystemStateStorage:
    """
    Persist process-coordination state in SQLite.

    Used by app.factory to invalidate process-local caches across workers/processes.
    """

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or db_base.SessionLocal
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


__all__ = ["SystemStateStorage"]
