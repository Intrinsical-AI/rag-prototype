from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_rag_backend.app.application import docs as docs_import


def test_execute_import_docs_sync_rejects_empty_payload() -> None:
    with pytest.raises(docs_import.ImportPayloadEmptyError):
        docs_import.execute_import_docs_sync(raw=b"", settings_obj="settings", ports="ports")


def test_execute_import_docs_sync_rejects_oversized_payload() -> None:
    with pytest.raises(docs_import.ImportFileTooLargeError):
        docs_import.execute_import_docs_sync(
            raw=b"x" * 11,
            settings_obj="settings",
            ports="ports",
            max_bytes=10,
        )


def test_execute_import_docs_sync_rejects_unsupported_format(monkeypatch) -> None:
    monkeypatch.setattr(
        docs_import,
        "detect_json_export_format",
        lambda _raw: SimpleNamespace(fmt="unknown"),
        raising=True,
    )
    with pytest.raises(docs_import.UnsupportedImportFormatError):
        docs_import.execute_import_docs_sync(raw=b"{}", settings_obj="settings", ports="ports")


def test_execute_import_docs_sync_parses_and_ingests_chatgpt(monkeypatch) -> None:
    class _Loader:
        def __init__(self, raw: bytes) -> None:
            assert raw == b"payload"

        def load(self):
            return [SimpleNamespace(text="a"), SimpleNamespace(text=" "), SimpleNamespace(text="b")]

    monkeypatch.setattr(
        docs_import,
        "detect_json_export_format",
        lambda _raw: SimpleNamespace(fmt="chatgpt_export"),
        raising=True,
    )
    monkeypatch.setattr(docs_import, "ChatGPTLoader", _Loader, raising=True)
    monkeypatch.setattr(
        docs_import.docs_service,
        "ingest_docs_sync",
        lambda *, texts, settings_obj, ports: [10, 11] if texts == ["a", "b"] else [],
        raising=True,
    )

    out = docs_import.execute_import_docs_sync(
        raw=b"payload",
        settings_obj="settings",
        ports="ports",
    )
    assert out.count == 2
    assert out.ids == [10, 11]
    assert out.input_texts == 2
    assert out.format_detected == "chatgpt_export"


def test_execute_import_docs_sync_maps_loader_value_error(monkeypatch) -> None:
    class _BrokenLoader:
        def __init__(self, raw: bytes) -> None:
            assert raw

        def load(self):
            raise ValueError("invalid export")

    monkeypatch.setattr(
        docs_import,
        "detect_json_export_format",
        lambda _raw: SimpleNamespace(fmt="gemini_export"),
        raising=True,
    )
    monkeypatch.setattr(docs_import, "GeminiLoader", _BrokenLoader, raising=True)

    with pytest.raises(docs_import.InvalidImportPayloadError, match="invalid export"):
        docs_import.execute_import_docs_sync(raw=b"payload", settings_obj="settings", ports="ports")
