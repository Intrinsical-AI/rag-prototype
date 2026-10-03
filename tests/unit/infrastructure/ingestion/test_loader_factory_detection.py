# tests/unit/infrastructure/ingestion/test_loader_factory_detection.py

from __future__ import annotations

from local_rag_backend.infrastructure.ingestion.loaders.factory import (
    detect_file_format,
    get_loader_for_file,
)
from local_rag_backend.infrastructure.ingestion.loaders.markdown_loader import MarkdownLoader
from local_rag_backend.infrastructure.ingestion.loaders.text_loader import TextFileLoader


def test_detect_file_format_known_text_extension_wins_over_csv_shape(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")

    det = detect_file_format(p, use_magic=False)
    assert det.fmt == "text"
    loader = get_loader_for_file(p, use_magic=False)
    assert loader is not None
    assert [item.text for item in loader.load()] == ["a;b\n1;2\n3;4\n"]


def test_detect_file_format_unknown_extension_requires_consistent_csv_columns(tmp_path):
    prose = tmp_path / "notes.unknown"
    prose.write_text("first, line\nsecond line\n", encoding="utf-8")
    assert detect_file_format(prose, use_magic=False).fmt == "text"

    table = tmp_path / "table.unknown"
    table.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
    assert detect_file_format(table, use_magic=False).fmt == "csv"


def test_detect_file_format_accepts_utf8_non_ascii_text(tmp_path):
    p = tmp_path / "nota.txt"
    p.write_text("¿Cómo está? Información útil.\nLínea 2.", encoding="utf-8")

    det = detect_file_format(p, use_magic=False)
    assert det.fmt == "text"
    assert get_loader_for_file(p, use_magic=False) is not None


def test_detect_file_format_known_extension_does_not_override_binary_guard(tmp_path):
    p = tmp_path / "unsafe.txt"
    p.write_bytes(b"first\x00second")
    assert detect_file_format(p, use_magic=False).fmt == "binary"


def test_detect_file_format_read_error_is_unknown(tmp_path, monkeypatch):
    p = tmp_path / "blocked.txt"
    p.write_text("hello", encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.ingestion.loaders.factory._read_head",
        _boom,
        raising=True,
    )

    det = detect_file_format(p, use_magic=False)
    assert det.fmt == "unknown"
    assert det.reason == "read-error"


def test_text_and_markdown_loaders_keep_distinct_lineage_and_format(tmp_path):
    raw = "# Title\n\n  Mixed Case  \n"
    supplied_metadata = {"owner": "editor"}
    for suffix, loader_type, expected_format in (
        ("txt", TextFileLoader, "text"),
        ("md", MarkdownLoader, "markdown"),
    ):
        path = tmp_path / f"note.{suffix}"
        path.write_text(raw, encoding="utf-8")
        [item] = loader_type(path, metadata=supplied_metadata).load()
        assert item.text == raw
        assert item.metadata == {
            "owner": "editor",
            "source_path": str(path),
            "filename": path.name,
            "format": expected_format,
        }
        assert item.lineage.loader_name == loader_type.__name__
        assert item.lineage.source_uri == str(path.resolve())
        assert item.lineage.record_locator == "file"
    assert supplied_metadata == {"owner": "editor"}
