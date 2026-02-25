"""Application use case for docs query/listing operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from local_rag_backend.infrastructure.persistence.sql.models import Document as DbDocument

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def list_docs_page_sync(*, db: Session, limit: int, offset: int) -> list[Any]:
    """List persisted documents using stable ascending ID order."""
    return cast(
        "list[Any]",
        db.query(DbDocument).order_by(DbDocument.doc_id.asc()).offset(offset).limit(limit).all(),
    )


__all__ = ["list_docs_page_sync"]
