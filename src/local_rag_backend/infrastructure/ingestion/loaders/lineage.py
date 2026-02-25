"""Lineage helpers for ingestion loaders."""

from __future__ import annotations

from local_rag_backend.core.domain.types import ItemLineage, TransformStep, utc_now


def loader_lineage(
    *,
    source_uri: str,
    loader_name: str,
    source_version: str | None = None,
    record_locator: str | None = None,
    offset_start: int | None = None,
    offset_end: int | None = None,
) -> ItemLineage:
    return ItemLineage(
        source_uri=source_uri,
        loader_name=loader_name,
        source_version=source_version,
        record_locator=record_locator,
        captured_at=utc_now(),
        offset_start=offset_start,
        offset_end=offset_end,
        transforms=(TransformStep(name="extract", version="v1", timestamp=utc_now()),),
    )
