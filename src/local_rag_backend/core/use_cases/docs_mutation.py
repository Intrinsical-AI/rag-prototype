"""Unified durable mutation coordinator for docs/index write operations."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from local_rag_backend.core.use_cases.docs_mutation_contracts import (
    MutationIntent,
    MutationUpsertInput,
    intent_to_dict,
    normalize_intent,
    summary_from_record,
)
from local_rag_backend.core.use_cases.docs_mutation_handlers import (
    apply_sql_mutation,
    apply_vector_delta,
    capture_before_image,
    recover_record,
    rollback_sql,
)
from local_rag_backend.core.use_cases.docs_mutation_runtime import (
    build_journal,
    mutation_uow_context,
    upsert_journal_record,
    uses_vector_index,
    validate_storage_profile,
    write_lock_context,
)
from local_rag_backend.core.use_cases.results import MutationSummary

if TYPE_CHECKING:
    from local_rag_backend.core.ports.contracts import DocsMutationPorts
    from local_rag_backend.settings import Settings


def _new_op_id() -> str:
    return f"mut:{uuid.uuid4().hex}"


class MutationCoordinator:
    def __init__(self, *, settings_obj: Settings, ports: DocsMutationPorts) -> None:
        self.settings_obj = settings_obj
        self.ports = ports

    def execute(self, intent: MutationIntent) -> MutationSummary:
        normalized = normalize_intent(intent=intent, new_op_id=_new_op_id)
        vector_mode_enabled = uses_vector_index(settings_obj=self.settings_obj)
        validate_storage_profile(
            settings_obj=self.settings_obj,
            ports=self.ports,
            vector_mode_enabled=vector_mode_enabled,
        )
        journal = build_journal(ports=self.ports)

        existing = journal.get(normalized.op_id)
        if existing is not None:
            if existing.state == "COMMITTED":
                return summary_from_record(existing)
            recover_record(
                journal=journal,
                record=existing,
                rollback_sql_fn=self._rollback_sql_with_uow,
                doc_repo_factory=self.ports.doc_repo_factory,
            )
            existing = journal.get(normalized.op_id)
            if existing is not None and existing.state == "COMMITTED":
                return summary_from_record(existing)

        with write_lock_context(settings_obj=self.settings_obj, ports=self.ports):
            existing_locked = journal.get(normalized.op_id)
            if existing_locked is not None and existing_locked.state == "COMMITTED":
                return summary_from_record(existing_locked)

            doc_repo = self.ports.doc_repo_factory()
            before_image = capture_before_image(doc_repo=doc_repo, intent=normalized)
            upsert_journal_record(
                journal=journal,
                op_id=normalized.op_id,
                state="PREPARED",
                intent=normalized,
                before_image=before_image,
            )

            with mutation_uow_context(ports=self.ports):
                sql_outcome = apply_sql_mutation(
                    doc_repo=doc_repo,
                    intent=normalized,
                    build_upsert_doc=self.ports.build_upsert_doc,
                )
            upsert_journal_record(
                journal=journal,
                op_id=normalized.op_id,
                state="SQL_COMMITTED",
                intent=normalized,
                before_image=before_image,
            )

            deleted_index: int | None = None
            index_doc_count: int | None = None
            try:
                if vector_mode_enabled:
                    deleted_index, index_doc_count = apply_vector_delta(
                        sql_outcome=sql_outcome,
                        doc_repo=doc_repo,
                        settings_obj=self.settings_obj,
                        ports=self.ports,
                        uses_vector_index=vector_mode_enabled,
                    )
                    upsert_journal_record(
                        journal=journal,
                        op_id=normalized.op_id,
                        state="VECTOR_COMMITTED",
                        intent=normalized,
                        before_image=before_image,
                    )
            except Exception as vector_err:
                upsert_journal_record(
                    journal=journal,
                    op_id=normalized.op_id,
                    state="COMPENSATING",
                    intent=normalized,
                    before_image=before_image,
                    error=str(vector_err),
                )
                try:
                    self._rollback_sql_with_uow(
                        doc_repo=doc_repo,
                        intent=intent_to_dict(normalized),
                        before_image=before_image,
                    )
                    upsert_journal_record(
                        journal=journal,
                        op_id=normalized.op_id,
                        state="ROLLED_BACK",
                        intent=normalized,
                        before_image=before_image,
                        error=str(vector_err),
                    )
                except Exception as rollback_err:
                    upsert_journal_record(
                        journal=journal,
                        op_id=normalized.op_id,
                        state="FAILED_NEEDS_RECOVERY",
                        intent=normalized,
                        before_image=before_image,
                        error=f"vector={vector_err}; rollback={rollback_err}",
                    )
                    raise RuntimeError(
                        "Mutation failed after SQL commit and rollback did not complete."
                    ) from rollback_err
                raise

            summary = MutationSummary(
                op_id=normalized.op_id,
                inserted=sql_outcome.inserted,
                updated=sql_outcome.updated,
                unchanged=sql_outcome.unchanged,
                deleted_sql=sql_outcome.deleted_sql,
                deleted_index=deleted_index,
                tombstoned=sql_outcome.tombstoned,
                missing_external_ids=list(sql_outcome.missing_external_ids),
                index_rebuilt=False,
                index_doc_count=index_doc_count,
                results=list(sql_outcome.results),
            )
            upsert_journal_record(
                journal=journal,
                op_id=normalized.op_id,
                state="COMMITTED",
                intent=normalized,
                before_image=before_image,
                outcome=summary,
            )
            journal.delete(normalized.op_id)
            return summary

    def recover_incomplete(self, *, limit: int = 100) -> int:
        journal = build_journal(ports=self.ports)
        records = journal.list_incomplete(limit=limit)
        if not records:
            return 0
        repaired = 0
        with write_lock_context(settings_obj=self.settings_obj, ports=self.ports):
            for record in records:
                recover_record(
                    journal=journal,
                    record=record,
                    rollback_sql_fn=self._rollback_sql_with_uow,
                    doc_repo_factory=self.ports.doc_repo_factory,
                )
                repaired += 1
        return repaired

    def _rollback_sql_with_uow(
        self,
        *,
        doc_repo: Any,
        intent: dict[str, Any],
        before_image: dict[str, Any],
    ) -> None:
        with mutation_uow_context(ports=self.ports):
            rollback_sql(doc_repo=doc_repo, intent=intent, before_image=before_image)


__all__ = ["MutationCoordinator", "MutationIntent", "MutationUpsertInput"]
