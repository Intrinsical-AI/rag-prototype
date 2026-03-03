from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from local_rag_backend.infrastructure.persistence.sql import base as db_base
from local_rag_backend.infrastructure.persistence.sql.models import Document as DbDocument


def test_chunk_dedup_unique_index_enforced(in_memory_sqlite):
    SessionLocal = db_base.SessionLocal
    s = SessionLocal()
    try:
        s.add(
            DbDocument(
                doc_id="doc:1",
                content="a",
                external_id="x1",
                chunk_dedup_sha256="same",
                content_sha256="h1",
                metadata_={},
            )
        )
        s.commit()

        s.add(
            DbDocument(
                doc_id="doc:2",
                content="b",
                external_id="x2",
                chunk_dedup_sha256="same",
                content_sha256="h2",
                metadata_={},
            )
        )
        with pytest.raises(IntegrityError):
            s.commit()
    finally:
        s.close()
