# src/local_rag_backend/infrastructure/ingestion/loaders/markdown_loader.py
"""
Markdown loader.

This currently reads the raw markdown as text. Chunking/formatting happens later
in the ingestion pipeline / CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.core.ports import LoaderPort
from local_rag_backend.infrastructure.ingestion.loaders.lineage import loader_lineage

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any


class MarkdownLoader(LoaderPort):
    def __init__(
        self,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.encoding = encoding
        self.errors = errors
        self._metadata = dict(metadata) if metadata else None

    def load(self) -> Iterable[LoadedItem]:
        text = self.path.read_text(encoding=self.encoding, errors=self.errors)
        md = dict(self._metadata) if self._metadata else {}
        md.setdefault("source_path", str(self.path))
        md.setdefault("filename", self.path.name)
        md.setdefault("format", "markdown")
        yield LoadedItem(
            text=text,
            lineage=loader_lineage(
                source_uri=str(self.path.resolve()),
                loader_name="MarkdownLoader",
                source_version=None,
                record_locator="file",
            ),
            metadata=md or None,
        )
