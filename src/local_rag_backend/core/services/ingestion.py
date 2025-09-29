# src/core/services/ingestion.py
"""
Ingestion service for document processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.utils import preprocess_text


def default_preprocess(text: str, _metadata: Mapping[str, Any] | None = None) -> str:
    """Default preprocessing function: lowercase, strip, and remove HTML tags."""
    return preprocess_text(text)


def default_chunker(
    max_chars: int = 1000, overlap: int = 100
) -> Callable[[str, Mapping[str, Any] | None], list[str]]:
    """Default chunker splitting text by character count with overlap."""

    # Ensure overlap is strictly less than max_chars
    safe_overlap = max(0, min(overlap, max_chars - 1))

    def _chunk(text: str, _metadata: Mapping[str, Any] | None = None) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += max_chars - safe_overlap
        return chunks

    return _chunk


def default_formatter(text: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Default formatter adding metadata as a header to the text."""
    if not metadata:
        return text
    header = "\n".join(f"{k.title()}: {v}" for k, v in metadata.items() if v is not None)
    return f"{header}\n\n{text}" if header else text


if TYPE_CHECKING:
    from local_rag_backend.core.ports import LoaderPort


class IngestionPipeline:
    """Pipeline for processing documents through loading, chunking, and storage."""

    def __init__(
        self,
        loader: LoaderPort,
        etl_service: ETLService,
        preprocess_fn: Callable[[str, Mapping[str, Any] | None], str] | None = None,
        chunk_fn: Callable[[str, Mapping[str, Any] | None], list[str]] | None = None,
        format_fn: Callable[[str, Mapping[str, Any] | None], str] | None = None,
    ) -> None:
        self.loader = loader
        self.etl_service = etl_service
        self.preprocess_fn = preprocess_fn or default_preprocess
        self.chunk_fn = chunk_fn or default_chunker()
        self.format_fn = format_fn or default_formatter

    def run(self) -> int:
        """Execute the ingestion pipeline."""
        all_chunks = []
        for loaded_item in self.loader.load():
            # Preprocess
            processed_text = self.preprocess_fn(loaded_item.text, loaded_item.metadata)

            # Chunk
            chunks = self.chunk_fn(processed_text, loaded_item.metadata)

            # Format each chunk
            for chunk in chunks:
                formatted_chunk = self.format_fn(chunk, loaded_item.metadata)
                all_chunks.append(formatted_chunk)

        # Ingest all chunks at once
        if all_chunks:
            self.etl_service.ingest(all_chunks)

        return len(all_chunks)
