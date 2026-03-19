from __future__ import annotations

import pytest

from local_rag_backend.core.use_cases.docs_import_canonical import (
    CanonicalImportDocumentInput,
    CanonicalImportRequestInput,
    execute_import_canonical_sync,
)


def test_execute_import_canonical_sync_rejects_blank_document_fields() -> None:
    with pytest.raises(ValueError, match=r"documents\[0\]\.external_id must not be blank"):
        execute_import_canonical_sync(
            request=CanonicalImportRequestInput(
                scope="repogpt:demo",
                snapshot_id="snap-1",
                replace_scope=True,
                documents=(
                    CanonicalImportDocumentInput(external_id="   ", content="alpha"),
                    CanonicalImportDocumentInput(external_id="doc-2", content="   "),
                ),
            ),
            settings_obj="settings",  # type: ignore[arg-type]
            ports="ports",  # type: ignore[arg-type]
        )


def test_execute_import_canonical_sync_rejects_mixed_valid_and_invalid_docs() -> None:
    with pytest.raises(ValueError, match=r"documents\[1\]\.content must not be blank"):
        execute_import_canonical_sync(
            request=CanonicalImportRequestInput(
                scope="repogpt:demo",
                snapshot_id="snap-2",
                replace_scope=True,
                documents=(
                    CanonicalImportDocumentInput(external_id="doc-1", content="alpha"),
                    CanonicalImportDocumentInput(external_id="doc-2", content="   "),
                ),
            ),
            settings_obj="settings",  # type: ignore[arg-type]
            ports="ports",  # type: ignore[arg-type]
        )
