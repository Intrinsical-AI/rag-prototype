"""Unit tests for ingestion.py pure helpers."""

from __future__ import annotations

from local_rag_backend.core.domain.types import ItemLineage, TransformStep
from local_rag_backend.core.services.ingestion import default_preprocess, stable_lineage_metadata

# ---------------------------------------------------------------------------
# default_preprocess
# ---------------------------------------------------------------------------


def test_default_preprocess_lowercases_and_strips() -> None:
    assert default_preprocess("  Hello World  ") == "hello world"


def test_default_preprocess_removes_html() -> None:
    assert default_preprocess("<b>Bold</b> text") == "bold text"


def test_default_preprocess_collapses_whitespace() -> None:
    assert default_preprocess("a   b\n\tc") == "a b c"


def test_default_preprocess_empty_string() -> None:
    assert default_preprocess("") == ""


def test_default_preprocess_ignores_metadata_arg() -> None:
    # metadata is accepted but has no effect on the output
    result = default_preprocess("Hello", {"key": "value"})
    assert result == "hello"


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
