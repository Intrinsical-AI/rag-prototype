# src/local_rag_backend/infrastructure/ingestion/loaders/text_loader.py
"""
Plain text loader (UTF-8 best-effort).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.core.ports import LoaderPort

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any


class TextFileLoader(LoaderPort):
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
        md.setdefault("format", "text")
        yield LoadedItem(text=text, metadata=md or None)
