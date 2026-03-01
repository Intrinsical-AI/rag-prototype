"""Application use case for importing JSON exports and ingesting their text payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from local_rag_backend.core.use_cases.docs_ingest import ingest_docs_sync

if TYPE_CHECKING:
    from local_rag_backend.core.ports import DocsImportLoaderPort
    from local_rag_backend.core.ports.contracts import DocsMutationPorts
    from local_rag_backend.settings import Settings

DEFAULT_IMPORT_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


class DocsImportError(ValueError):
    """Base error for docs-import use case failures."""


class ImportFileTooLargeError(DocsImportError):
    def __init__(self, *, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
        super().__init__(
            f"File too large. Maximum allowed size is {self.max_bytes // (1024 * 1024)} MB."
        )


class ImportPayloadEmptyError(DocsImportError):
    def __init__(self) -> None:
        super().__init__("Uploaded file is empty.")


class UnsupportedImportFormatError(DocsImportError):
    def __init__(self) -> None:
        super().__init__(
            "Could not detect a supported export format. "
            "Expected a ChatGPT or Google Gemini JSON export."
        )


class InvalidImportPayloadError(DocsImportError):
    """Input payload is a supported format but malformed."""


@dataclass(frozen=True)
class ImportDocsOutcome:
    count: int
    ids: list[str]
    format_detected: str
    input_texts: int


def execute_import_docs_sync(
    *,
    raw: bytes,
    settings_obj: Settings,
    ports: DocsMutationPorts,
    import_loader: DocsImportLoaderPort,
    max_bytes: int = DEFAULT_IMPORT_MAX_BYTES,
) -> ImportDocsOutcome:
    if len(raw) > int(max_bytes):
        raise ImportFileTooLargeError(max_bytes=max_bytes)

    if not raw:
        raise ImportPayloadEmptyError()

    try:
        load_result = import_loader.load_texts(raw=raw)
    except UnsupportedImportFormatError:
        raise
    except ValueError as exc:
        raise InvalidImportPayloadError(str(exc)) from exc

    texts = [text for text in load_result.texts if text and text.strip()]
    if not texts:
        return ImportDocsOutcome(
            count=0,
            ids=[],
            format_detected=load_result.format_detected,
            input_texts=0,
        )

    ids = ingest_docs_sync(
        texts=texts,
        settings_obj=settings_obj,
        ports=ports,
        source="api:/docs/import",
    )
    return ImportDocsOutcome(
        count=len(ids),
        ids=ids,
        format_detected=load_result.format_detected,
        input_texts=len(texts),
    )


__all__ = [
    "DEFAULT_IMPORT_MAX_BYTES",
    "DocsImportError",
    "ImportDocsOutcome",
    "ImportFileTooLargeError",
    "ImportPayloadEmptyError",
    "InvalidImportPayloadError",
    "UnsupportedImportFormatError",
    "execute_import_docs_sync",
]
