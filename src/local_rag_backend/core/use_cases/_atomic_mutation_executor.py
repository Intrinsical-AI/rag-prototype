"""Atomic mutation executor for unified backends such as Elasticsearch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.use_cases.results import MutationSummary, UpsertDocResult

if TYPE_CHECKING:
    from local_rag_backend.core.ports.contracts import DocsMutationPorts
    from local_rag_backend.core.use_cases._mutation_saga_executor import PreparedMutation
    from local_rag_backend.settings import Settings


class AtomicMutationExecutor:
    def __init__(self, *, settings_obj: Settings, ports: DocsMutationPorts) -> None:
        self.settings_obj = settings_obj
        self.ports = ports

    def execute_locked(self, *, prepared: PreparedMutation) -> MutationSummary:
        intent = prepared.intent
        doc_repo = self.ports.doc_repo_factory()
        inserted = 0
        updated = 0
        unchanged = 0
        deleted_sql = 0
        deleted_index = 0 if prepared.vector_mode_enabled else None
        tombstoned = 0
        missing_external_ids: list[str] = []
        results: list[UpsertDocResult] = []

        if intent.upserts:
            delete_tombstones = getattr(doc_repo, "delete_tombstones", None)
            if callable(delete_tombstones):
                delete_tombstones([u.external_id for u in intent.upserts])

            items = [
                self.ports.build_upsert_doc(
                    external_id=u.external_id,
                    content=u.content,
                    source_id=u.source_id,
                    scope=u.scope,
                    snapshot_id=u.snapshot_id,
                    metadata=u.metadata,
                    embedding=prepared.precomputed_vectors_by_external_id.get(u.external_id),
                )
                for u in intent.upserts
            ]
            upsert_results, _changed, _updated_ids = doc_repo.upsert_documents_by_external_id(items)
            for row in upsert_results:
                action = str(row.action)
                if action == "inserted":
                    inserted += 1
                elif action == "updated":
                    updated += 1
                else:
                    unchanged += 1
                results.append(
                    UpsertDocResult(
                        external_id=str(row.external_id),
                        id=str(row.id),
                        action=action,
                        content_changed=bool(row.content_changed),
                    )
                )

        deleted_doc_ids: list[DocId] = []
        if intent.delete_ids:
            ids_doc = [DocId(str(x)) for x in intent.delete_ids]
            existing_docs = list(doc_repo.get(ids_doc))
            deleted_sql += len(existing_docs)
            deleted_doc_ids.extend([DocId(str(d.id)) for d in existing_docs])
            if existing_docs:
                doc_repo.delete_documents(ids_doc)

        if intent.delete_external_ids:
            deleted_count, deleted_ids, missing, tombstoned_count = doc_repo.delete_by_external_ids(
                list(intent.delete_external_ids)
            )
            deleted_sql += int(deleted_count)
            deleted_doc_ids.extend([DocId(str(x)) for x in deleted_ids])
            missing_external_ids.extend([str(x) for x in missing])
            tombstoned += int(tombstoned_count)

        if prepared.vector_mode_enabled and deleted_index is not None:
            deleted_index = len({str(doc_id) for doc_id in deleted_doc_ids})

        index_doc_count = None
        if prepared.vector_mode_enabled and hasattr(doc_repo, "get_all_documents"):
            index_doc_count = len(list(doc_repo.get_all_documents()))

        return MutationSummary(
            op_id=intent.op_id,
            inserted=inserted,
            updated=updated,
            unchanged=unchanged,
            deleted_sql=deleted_sql,
            deleted_index=deleted_index,
            tombstoned=tombstoned,
            missing_external_ids=missing_external_ids,
            index_rebuilt=False,
            index_doc_count=index_doc_count,
            results=results,
        )

    def recover_incomplete(self, *, limit: int = 100) -> int:
        _ = limit
        return 0


__all__ = ["AtomicMutationExecutor"]
