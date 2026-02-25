"""Application orchestration for docs listing/import flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from local_rag_backend.app.services import docs as docs_service
from local_rag_backend.infrastructure.ingestion.loaders import (
    ChatGPTLoader,
    GeminiLoader,
    detect_json_export_format,
)
from local_rag_backend.infrastructure.persistence.sql.models import Document as DbDocument

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from local_rag_backend.app.services.ports import DocsMutationPorts
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


def list_docs_page_sync(*, db: Session, limit: int, offset: int) -> list[Any]:
    """List persisted documents using stable ascending ID order."""
    return cast(
        "list[Any]",
        db.query(DbDocument).order_by(DbDocument.doc_id.asc()).offset(offset).limit(limit).all(),
    )


def execute_import_docs_sync(
    *,
    raw: bytes,
    settings_obj: Settings,
    ports: DocsMutationPorts,
    max_bytes: int = DEFAULT_IMPORT_MAX_BYTES,
) -> ImportDocsOutcome:
    """Validate/export-parse/import documents from bytes payload."""
    if len(raw) > int(max_bytes):
        raise ImportFileTooLargeError(max_bytes=max_bytes)

    if not raw:
        raise ImportPayloadEmptyError()

    detection = detect_json_export_format(raw)
    if detection.fmt == "chatgpt_export":
        loader: ChatGPTLoader | GeminiLoader = ChatGPTLoader(raw)
    elif detection.fmt == "gemini_export":
        loader = GeminiLoader(raw)
    else:
        raise UnsupportedImportFormatError()

    try:
        items = list(loader.load())
    except ValueError as exc:
        raise InvalidImportPayloadError(str(exc)) from exc

    texts = [item.text for item in items if item.text and item.text.strip()]
    if not texts:
        return ImportDocsOutcome(count=0, ids=[], format_detected=detection.fmt, input_texts=0)

    ids = docs_service.ingest_docs_sync(
        texts=texts,
        settings_obj=settings_obj,
        ports=ports,
    )
    return ImportDocsOutcome(
        count=len(ids),
        ids=ids,
        format_detected=detection.fmt,
        input_texts=len(texts),
    )
