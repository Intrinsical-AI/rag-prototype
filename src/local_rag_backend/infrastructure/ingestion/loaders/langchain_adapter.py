# src/local_rag_backend/infrastructure/ingestion/loaders/langchain_adapter.py
"""
Adapter to use LangChain loaders as `LoaderPort` implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.core.ports import LoaderPort
from local_rag_backend.infrastructure.ingestion.loaders.lineage import loader_lineage

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class LangChainLoader(LoaderPort):
    """Wrap a LangChain document loader into our `LoaderPort`."""

    def __init__(
        self,
        lc_loader: Any,
        *,
        drop_empty: bool = True,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> None:
        self._loader = lc_loader
        self._drop_empty = drop_empty
        self._metadata_filter = dict(metadata_filter) if metadata_filter else None

    def load(self) -> Iterable[LoadedItem]:
        docs = self._loader.load()

        for i, doc in enumerate(docs):
            text, metadata = _extract_text_and_metadata(doc)

            if self._drop_empty and (not text or not text.strip()):
                continue

            if self._metadata_filter:
                md = metadata or {}
                if any(md.get(k) != v for k, v in self._metadata_filter.items()):
                    continue

            source_uri = "langchain://loader"
            locator = f"item:{i}"
            if metadata:
                source_uri = str(
                    metadata.get("source")
                    or metadata.get("url")
                    or metadata.get("path")
                    or source_uri
                )
                locator = str(metadata.get("id") or locator)

            yield LoadedItem(
                text=text,
                lineage=loader_lineage(
                    source_uri=source_uri,
                    loader_name="LangChainLoader",
                    source_version="langchain-loader",
                    record_locator=locator,
                ),
                metadata=metadata,
            )


def _extract_text_and_metadata(doc: Any) -> tuple[str, Mapping[str, Any] | None]:
    """Extract `(text, metadata)` from a LangChain `Document` or similar object."""
    text: str | None = None
    metadata: Mapping[str, Any] | None = None

    if hasattr(doc, "page_content"):
        text = doc.page_content
    if hasattr(doc, "metadata"):
        metadata = doc.metadata

    if text is None and isinstance(doc, dict):
        maybe = doc.get("page_content")
        if isinstance(maybe, str):
            text = maybe
        metadata_val = doc.get("metadata")
        if isinstance(metadata_val, dict):
            metadata = metadata_val

    if text is None:
        text = str(doc)

    return text, metadata
