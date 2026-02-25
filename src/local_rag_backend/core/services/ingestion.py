# src/core/services/ingestion.py
"""
Ingestion service for document processing.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from local_rag_backend.core.domain.types import ItemLineage
    from local_rag_backend.core.ports import LoaderPort
    from local_rag_backend.core.services.etl import ETLService
    from local_rag_backend.settings import Settings
from local_rag_backend.core.domain.types import TransformStep, utc_now
from local_rag_backend.core.services.chunking import chunk_chars_v1
from local_rag_backend.core.services.text_processing import preprocess_text


def default_preprocess(text: str, _metadata: Mapping[str, Any] | None = None) -> str:
    """Default preprocessing function: lowercase, strip, and remove HTML tags."""
    return preprocess_text(text)


def default_chunker(
    max_chars: int = 1000, overlap: int = 100
) -> Callable[[str, Mapping[str, Any] | None], list[str]]:
    """Default chunker splitting text by character count with overlap."""

    def _chunk(text: str, _metadata: Mapping[str, Any] | None = None) -> list[str]:
        return [c.text for c in chunk_chars_v1(text, max_chars=max_chars, overlap=overlap)]

    return _chunk


def default_formatter(text: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Default formatter adding metadata as a header to the text."""
    if not metadata:
        return text
    header = "\n".join(
        f"{k.title()}: {v}"
        for k, v in metadata.items()
        if v is not None and not str(k).startswith("_")
    )
    return f"{header}\n\n{text}" if header else text


def build_preprocess_fn_from_settings(
    settings: Settings,
) -> Callable[[str, Mapping[str, Any] | None], str]:
    def _fn(text: str, _metadata: Mapping[str, Any] | None = None) -> str:
        return preprocess_text(
            text,
            lowercase=settings.ingest_clean_lowercase,
            remove_html=settings.ingest_clean_remove_html,
            collapse_whitespace=settings.ingest_clean_collapse_whitespace,
            strip=settings.ingest_clean_strip,
        )

    return _fn


def build_chunk_fn_from_settings(
    settings: Settings,
) -> Callable[[str, Mapping[str, Any] | None], list[str]]:
    if settings.ingest_chunk_strategy != "chars_v1":
        raise ValueError(f"Unsupported ingest_chunk_strategy: {settings.ingest_chunk_strategy!r}")
    return default_chunker(settings.ingest_chunk_chars, settings.ingest_chunk_overlap)


def _with_transform(
    lineage: ItemLineage, *, name: str, version: str, params: dict[str, Any]
) -> ItemLineage:
    step = TransformStep(name=name, version=version, params=params, timestamp=utc_now())
    return replace(lineage, transforms=(*lineage.transforms, step))


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
            preprocess_lineage = _with_transform(
                loaded_item.lineage,
                name="preprocess",
                version="v1",
                params={},
            )
            processed_text = self.preprocess_fn(loaded_item.text, loaded_item.metadata)

            chunk_lineage = _with_transform(
                preprocess_lineage,
                name="chunk",
                version="chars_v1",
                params={},
            )
            chunks = self.chunk_fn(processed_text, loaded_item.metadata)

            for chunk in chunks:
                format_lineage = _with_transform(
                    chunk_lineage,
                    name="format",
                    version="v1",
                    params={},
                )
                metadata = dict(loaded_item.metadata) if loaded_item.metadata else {}
                metadata["_lineage"] = asdict(format_lineage)
                formatted_chunk = self.format_fn(chunk, metadata)
                all_chunks.append(formatted_chunk)

        if all_chunks:
            self.etl_service.ingest(all_chunks)

        return len(all_chunks)
