# tests/unit/infrastructure/ingestion/test_loader_factory_detection.py

from __future__ import annotations

from local_rag_backend.infrastructure.ingestion.loaders.factory import detect_file_format


def test_detect_file_format_does_not_trust_extension_only(tmp_path):
    # Extension says "text", but content looks like CSV: we should detect csv via heuristics.
    p = tmp_path / "data.txt"
    p.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")

    det = detect_file_format(p, use_magic=False)
    assert det.fmt == "csv"
