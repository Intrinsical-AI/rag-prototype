"""Application use case for docs query/listing operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_rag_backend.core.ports import DocsReadPort, ListedDocument


def list_docs_page_sync(
    *,
    docs_reader: DocsReadPort,
    limit: int,
    offset: int,
) -> tuple[ListedDocument, ...]:
    """List persisted documents using stable ascending ID order."""
    return docs_reader.list_docs_page(limit=limit, offset=offset)


__all__ = ["list_docs_page_sync"]
