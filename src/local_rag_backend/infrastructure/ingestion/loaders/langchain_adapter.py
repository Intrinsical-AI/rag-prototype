# src/local_rag_backend/infrastructure/ingestion/loaders/langchain_adapter.py
"""
Adapter to use LangChain loaders as `LoaderPort` implementations.

Usage example (requires optional extra `loaders`):

    from langchain_community.document_loaders import WebBaseLoader
    from local_rag_backend.infrastructure.ingestion.loaders import LangChainLoader

    lc_loader = WebBaseLoader(["https://example.com"])  # any LangChain loader instance
    loader = LangChainLoader(lc_loader)

    pipeline = IngestionPipeline(loader=loader, etl_service=etl)
    pipeline.run()

This adapter avoids importing LangChain types at import time to remain optional.
It expects the provided loader instance to implement a `.load()` method returning
LangChain `Document` objects (with `.page_content` and `.metadata`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.core.ports import LoaderPort

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class LangChainLoader(LoaderPort):
    """Wrap a LangChain document loader into our `LoaderPort`.

    The adapter is intentionally permissive on types to keep `langchain-community`
    as an optional dependency. It relies on duck-typing for the returned objects.
    """

    def __init__(
        self,
        lc_loader: Any,
        *,
        drop_empty: bool = True,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            lc_loader: An instance of a LangChain loader (e.g., WebBaseLoader, DirectoryLoader,
                SitemapLoader, UnstructuredFileLoader, etc.). Must have a `.load()` method.
            drop_empty: If True, skip documents with empty/whitespace-only content.
            metadata_filter: If provided, only yield items whose metadata includes these
                key/value pairs (exact match).
        """
        self._loader = lc_loader
        self._drop_empty = drop_empty
        self._metadata_filter = dict(metadata_filter) if metadata_filter else None

    def load(self) -> Iterable[LoadedItem]:
        # Call the underlying LangChain loader
        docs = self._loader.load()

        # Support generators or lists
        for doc in docs:
            text, metadata = _extract_text_and_metadata(doc)

            if self._drop_empty and (not text or not text.strip()):
                continue

            if self._metadata_filter:
                md = metadata or {}
                if any(md.get(k) != v for k, v in self._metadata_filter.items()):
                    continue

            yield LoadedItem(text=text, metadata=metadata)


def _extract_text_and_metadata(doc: Any) -> tuple[str, Mapping[str, Any] | None]:
    """Extract `(text, metadata)` from a LangChain `Document` or similar object.

    - Prefers `doc.page_content` and `doc.metadata` attributes
    - Falls back to dict-like access if the loader returns plain dicts
    - As a last resort, converts the object to `str` for the text
    """
    text: str | None = None
    metadata: Mapping[str, Any] | None = None

    # Attribute access (most common for LangChain Document)
    if hasattr(doc, "page_content"):
        text = doc.page_content
    if hasattr(doc, "metadata"):
        metadata = doc.metadata

    # Fallback to dict-like access, if applicable
    if text is None and isinstance(doc, dict):
        maybe = doc.get("page_content")
        if isinstance(maybe, str):
            text = maybe
        metadata_val = doc.get("metadata")
        if isinstance(metadata_val, dict):
            metadata = metadata_val

    # Last resort: stringification
    if text is None:
        text = str(doc)

    return text, metadata
