"""Unit tests for ingestion.py pure helpers."""

from __future__ import annotations

from local_rag_backend.core.domain.types import ItemLineage, TransformStep
from local_rag_backend.core.services.ingestion import stable_lineage_metadata

# ---------------------------------------------------------------------------
# stable_lineage_metadata
# ---------------------------------------------------------------------------


def _lineage(**kwargs) -> ItemLineage:
    defaults = {"source_uri": "file:///a.txt", "loader_name": "plain_text"}
    return ItemLineage(**{**defaults, **kwargs})


def test_stable_lineage_metadata_basic_fields() -> None:
    meta = stable_lineage_metadata(_lineage())
    assert meta["source_uri"] == "file:///a.txt"
    assert meta["loader_name"] == "plain_text"
    assert meta["source_version"] is None
    assert meta["record_locator"] is None
    assert meta["offset_start"] is None
    assert meta["offset_end"] is None
    assert meta["transforms"] == []


def test_stable_lineage_metadata_optional_fields() -> None:
    lineage = _lineage(
        source_version="abc123",
        record_locator="row:42",
        offset_start=0,
        offset_end=100,
    )
    meta = stable_lineage_metadata(lineage)
    assert meta["source_version"] == "abc123"
    assert meta["record_locator"] == "row:42"
    assert meta["offset_start"] == 0
    assert meta["offset_end"] == 100


def test_stable_lineage_metadata_with_transforms() -> None:
    step = TransformStep(name="chunk", version="chars_v1", params={"max_chars": 500})
    lineage = _lineage(transforms=(step,))
    meta = stable_lineage_metadata(lineage)
    assert len(meta["transforms"]) == 1
    t = meta["transforms"][0]
    assert t["name"] == "chunk"
    assert t["version"] == "chars_v1"
    assert t["params"] == {"max_chars": 500}


def test_stable_lineage_metadata_transform_none_params() -> None:
    step = TransformStep(name="preprocess", version="v1", params=None)
    lineage = _lineage(transforms=(step,))
    meta = stable_lineage_metadata(lineage)
    assert meta["transforms"][0]["params"] is None


def test_stable_lineage_metadata_is_deterministic() -> None:
    lineage = _lineage(source_version="v1")
    assert stable_lineage_metadata(lineage) == stable_lineage_metadata(lineage)
