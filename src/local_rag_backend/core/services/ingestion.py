"""Stable lineage metadata for document ingestion."""

from __future__ import annotations

from typing import Any

from local_rag_backend.core.domain.types import ItemLineage


def stable_lineage_metadata(lineage: ItemLineage) -> dict[str, Any]:
    """Build deterministic lineage metadata without runtime timestamps."""
    return {
        "source_uri": str(lineage.source_uri),
        "loader_name": str(lineage.loader_name),
        "source_version": (
            str(lineage.source_version) if lineage.source_version is not None else None
        ),
        "record_locator": (
            str(lineage.record_locator) if lineage.record_locator is not None else None
        ),
        "offset_start": (int(lineage.offset_start) if lineage.offset_start is not None else None),
        "offset_end": (int(lineage.offset_end) if lineage.offset_end is not None else None),
        "transforms": [
            {
                "name": str(step.name),
                "version": str(step.version),
                "params": (dict(step.params) if step.params is not None else None),
            }
            for step in lineage.transforms
        ],
    }
