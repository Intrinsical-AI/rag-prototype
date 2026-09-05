"""Canonical import/sync use case for external document producers."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.use_cases.docs_mutation import (
    MutationCoordinator,
    MutationIntent,
    MutationUpsertInput,
    uses_vector_index,
)
from local_rag_backend.core.use_cases.errors import IndexRebuildRequiredError
from local_rag_backend.core.use_cases.results import CanonicalImportSummary, UpsertDocResult

if TYPE_CHECKING:
    from local_rag_backend.core.ports.contracts import DocsMutationPorts
    from local_rag_backend.settings import Settings


CANONICAL_BATCH_SIZE = 256


@dataclass(frozen=True)
class CanonicalImportDocumentInput:
    external_id: str
    content: str
    source_id: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CanonicalImportRequestInput:
    scope: str
    snapshot_id: str
    replace_scope: bool
    documents: tuple[CanonicalImportDocumentInput, ...]
    source: str = "unknown"


def execute_import_canonical_sync(
    *,
    request: CanonicalImportRequestInput,
    settings_obj: Settings,
    ports: DocsMutationPorts,
) -> CanonicalImportSummary:
    scope = str(request.scope).strip()
    snapshot_id = str(request.snapshot_id).strip()
    if not scope:
        raise ValueError("scope must not be blank")
    if not snapshot_id:
        raise ValueError("snapshot_id must not be blank")

    documents = _normalize_documents(
        documents=request.documents,
        scope=scope,
        snapshot_id=snapshot_id,
    )
    if not documents:
        raise ValueError("documents must not be empty")

    coordinator = MutationCoordinator(settings_obj=settings_obj, ports=ports)
    inserted = 0
    updated = 0
    unchanged = 0
    results: list[UpsertDocResult] = []

    for batch_idx, batch in enumerate(_batched(documents, size=CANONICAL_BATCH_SIZE), start=1):
        summary = coordinator.execute(
            MutationIntent(
                op_id=f"canon:{uuid.uuid4().hex}:batch:{batch_idx}",
                upserts=tuple(batch),
                source=request.source,
            )
        )
        inserted += int(summary.inserted)
        updated += int(summary.updated)
        unchanged += int(summary.unchanged)
        results.extend(list(summary.results or []))

    deleted_sql = 0
    deleted_index: int | None = 0 if uses_vector_index(settings_obj=settings_obj) else None
    stale_external_ids: list[str] = []
    if request.replace_scope:
        stale_external_ids, deleted_sql, deleted_index = _delete_stale_scope_documents(
            scope=scope,
            keep_external_ids={doc.external_id for doc in documents},
            settings_obj=settings_obj,
            ports=ports,
        )

    return CanonicalImportSummary(
        scope=scope,
        snapshot_id=snapshot_id,
        replace_scope=bool(request.replace_scope),
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        deleted_sql=deleted_sql,
        deleted_index=deleted_index,
        deleted_external_ids=stale_external_ids,
        results=results,
    )


def _normalize_documents(
    *,
    documents: Sequence[CanonicalImportDocumentInput],
    scope: str,
    snapshot_id: str,
) -> tuple[MutationUpsertInput, ...]:
    upserts: list[MutationUpsertInput] = []
    seen_external_ids: set[str] = set()
    validation_errors: list[str] = []
    duplicate_external_ids: list[str] = []

    for idx, document in enumerate(documents):
        external_id = str(document.external_id).strip()
        content = str(document.content).strip()

        if not external_id:
            validation_errors.append(f"documents[{idx}].external_id must not be blank")
        if not content:
            validation_errors.append(f"documents[{idx}].content must not be blank")
        if not external_id or not content:
            continue
        if external_id in seen_external_ids:
            duplicate_external_ids.append(external_id)
            continue
        seen_external_ids.add(external_id)
        metadata = dict(document.metadata) if document.metadata is not None else {}
        metadata["scope"] = scope
        metadata["snapshot_id"] = snapshot_id
        upserts.append(
            MutationUpsertInput(
                external_id=external_id,
                content=content,
                source_id=(
                    str(document.source_id).strip()
                    if document.source_id is not None and str(document.source_id).strip()
                    else None
                ),
                scope=scope,
                snapshot_id=snapshot_id,
                metadata=metadata,
            )
        )

    if duplicate_external_ids:
        validation_errors.append(
            "documents.external_id values must be unique per canonical import: "
            + ", ".join(sorted(set(duplicate_external_ids))[:10])
        )
    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    return tuple(upserts)


def _batched(
    documents: Sequence[MutationUpsertInput],
    *,
    size: int,
) -> Sequence[tuple[MutationUpsertInput, ...]]:
    return tuple(
        tuple(documents[idx : idx + size]) for idx in range(0, len(documents), max(1, int(size)))
    )


def _delete_stale_scope_documents(
    *,
    scope: str,
    keep_external_ids: set[str],
    settings_obj: Settings,
    ports: DocsMutationPorts,
) -> tuple[list[str], int, int | None]:
    colocated = settings_obj.persistence_backend == "elasticsearch"
    uow_factory = ports.mutation_uow_factory if not colocated else None
    if not colocated and not callable(uow_factory):
        raise RuntimeError("Canonical scope deletion requires a SQL mutation unit of work.")
    uow = uow_factory() if uow_factory is not None else nullcontext()
    vector_attempted = False
    try:
        with uow:
            doc_repo = ports.doc_repo_factory()
            stale_external_ids, stale_doc_ids = _resolve_stale_scope_documents(
                doc_repo=doc_repo,
                scope=scope,
                keep_external_ids=keep_external_ids,
                require_doc_ids=uses_vector_index(settings_obj=settings_obj) and not colocated,
            )
            if not stale_external_ids:
                return [], 0, (0 if uses_vector_index(settings_obj=settings_obj) else None)
            hard_delete = getattr(doc_repo, "hard_delete_by_external_ids", None)
            if not callable(hard_delete):
                raise RuntimeError(
                    "Configured document repository does not support hard_delete_by_external_ids."
                )
            apply_delta = None
            if uses_vector_index(settings_obj=settings_obj) and not colocated:
                vec_repo = ports.vector_repo_factory(
                    index_path=settings_obj.index_path,
                    id_map_path=settings_obj.id_map_path,
                    backend=settings_obj.vector_backend,
                    settings_obj=settings_obj,
                )
                apply_delta = getattr(vec_repo, "apply_delta_atomic", None)
                if not callable(apply_delta):
                    raise RuntimeError("Canonical scope deletion requires apply_delta_atomic.")
            deleted_raw = hard_delete(stale_external_ids)
            deleted = int(deleted_raw) if deleted_raw is not None else len(stale_external_ids)
            deleted_index: int | None = None
            if apply_delta is not None:
                vector_attempted = True
                apply_delta(delete_ids=stale_doc_ids, upserts=[])
                deleted_index = len(stale_doc_ids)
            elif colocated and uses_vector_index(settings_obj=settings_obj):
                # Elasticsearch stores the embedding in the document that was just deleted.
                deleted_index = deleted
        return stale_external_ids, deleted, deleted_index
    except Exception as exc:
        # Adapter failures and SQL commit failures both need the same explicit recovery path.
        # The existing vector delta can fail between saving the index and its ID map.
        if vector_attempted:
            raise IndexRebuildRequiredError() from exc
        raise


def _resolve_stale_scope_documents(
    *,
    doc_repo: Any,
    scope: str,
    keep_external_ids: set[str],
    require_doc_ids: bool,
) -> tuple[list[str], list[DocId]]:
    list_external_ids_by_scope = getattr(doc_repo, "list_external_ids_by_scope", None)
    if not callable(list_external_ids_by_scope):
        raise RuntimeError("Configured document repository does not support scope-aware sync.")

    existing_external_ids = {
        str(ext_id).strip()
        for ext_id in cast("Sequence[str]", list_external_ids_by_scope(scope))
        if str(ext_id).strip()
    }
    stale_external_ids = sorted(existing_external_ids - keep_external_ids)
    if not stale_external_ids:
        return [], []

    snapshot_by_external_ids = getattr(doc_repo, "snapshot_by_external_ids", None)
    stale_doc_ids: list[DocId] = []
    if require_doc_ids:
        if not callable(snapshot_by_external_ids):
            raise RuntimeError("Canonical scope deletion requires snapshot_by_external_ids.")
        snapshots = cast("Sequence[dict[str, Any]]", snapshot_by_external_ids(stale_external_ids))
        stale_doc_ids = [
            DocId(str(snapshot.get("id")))
            for snapshot in snapshots
            if str(snapshot.get("id") or "").strip()
        ]
        if len(set(stale_doc_ids)) != len(stale_external_ids):
            raise RuntimeError("Cannot resolve every stale document ID before deleting its vector.")
    return stale_external_ids, stale_doc_ids


__all__ = [
    "CANONICAL_BATCH_SIZE",
    "CanonicalImportDocumentInput",
    "CanonicalImportRequestInput",
    "execute_import_canonical_sync",
]
