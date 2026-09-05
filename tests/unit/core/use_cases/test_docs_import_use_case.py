from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_rag_backend.core.use_cases import docs_import as docs_import


def test_execute_import_docs_sync_rejects_empty_payload() -> None:
    class DummyImportLoader:
        def load_texts(self, *, raw: bytes):
            return SimpleNamespace(format_detected="chatgpt_export", texts=("x",))

    with pytest.raises(docs_import.ImportPayloadEmptyError):
        docs_import.execute_import_docs_sync(
            raw=b"",
            settings_obj="settings",
            ports="ports",
            import_loader=DummyImportLoader(),
        )


def test_execute_import_docs_sync_rejects_oversized_payload() -> None:
    class DummyImportLoader:
        def load_texts(self, *, raw: bytes):
            return SimpleNamespace(format_detected="chatgpt_export", texts=("x",))

    with pytest.raises(docs_import.ImportFileTooLargeError):
        docs_import.execute_import_docs_sync(
            raw=b"x" * 11,
            settings_obj="settings",
            ports="ports",
            import_loader=DummyImportLoader(),
            max_bytes=10,
        )


def test_execute_import_docs_sync_rejects_unsupported_format() -> None:
    class UnsupportedLoader:
        def load_texts(self, *, raw: bytes):
            raise docs_import.UnsupportedImportFormatError()

    with pytest.raises(docs_import.UnsupportedImportFormatError):
        docs_import.execute_import_docs_sync(
            raw=b"{}",
            settings_obj="settings",
            ports="ports",
            import_loader=UnsupportedLoader(),
        )


def test_execute_import_docs_sync_parses_and_ingests_chatgpt(monkeypatch) -> None:
    class DummyImportLoader:
        def load_texts(self, *, raw: bytes):
            assert raw == b"payload"
            return SimpleNamespace(format_detected="chatgpt_export", texts=("a", " ", "b"))

    monkeypatch.setattr(
        docs_import,
        "ingest_docs_sync",
        lambda *, texts, settings_obj, ports, source="api:/docs/import-conversations": (
            [10, 11] if texts == ["a", "b"] else []
        ),
        raising=True,
    )

    out = docs_import.execute_import_docs_sync(
        raw=b"payload",
        settings_obj="settings",
        ports="ports",
        import_loader=DummyImportLoader(),
    )
    assert out.count == 2
    assert out.ids == [10, 11]
    assert out.input_texts == 2
    assert out.format_detected == "chatgpt_export"


def test_execute_import_docs_sync_maps_loader_value_error() -> None:
    class BrokenLoader:
        def load_texts(self, *, raw: bytes):
            raise ValueError("invalid export")

    with pytest.raises(docs_import.InvalidImportPayloadError, match="invalid export"):
        docs_import.execute_import_docs_sync(
            raw=b"payload",
            settings_obj="settings",
            ports="ports",
            import_loader=BrokenLoader(),
        )
