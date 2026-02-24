# tests/unit/infrastructure/ingestion/test_loader_factory_detection.py

from __future__ import annotations

from local_rag_backend.infrastructure.ingestion.loaders.factory import (
    detect_file_format,
    get_loader_for_file,
)


def test_detect_file_format_does_not_trust_extension_only(tmp_path):
    # Extension says "text", but content looks like CSV: we should detect csv via heuristics.
    p = tmp_path / "data.txt"
    p.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")

    det = detect_file_format(p, use_magic=False)
    assert det.fmt == "csv"


def test_detect_file_format_accepts_utf8_non_ascii_text(tmp_path):
    p = tmp_path / "nota.txt"
    p.write_text("¿Cómo está? Información útil.\nLínea 2.", encoding="utf-8")

    det = detect_file_format(p, use_magic=False)
    assert det.fmt == "text"
    assert get_loader_for_file(p, use_magic=False) is not None


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
