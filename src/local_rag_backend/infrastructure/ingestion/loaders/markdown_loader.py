"""Load raw Markdown text with Markdown lineage metadata."""

from __future__ import annotations

from local_rag_backend.infrastructure.ingestion.loaders.text_loader import TextFileLoader


class MarkdownLoader(TextFileLoader):
    _format = "markdown"
    _loader_name = "MarkdownLoader"
