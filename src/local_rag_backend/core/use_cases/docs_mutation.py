"""Unified durable mutation coordinator for docs/index write operations."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from local_rag_backend.core.use_cases._mutation_saga_executor import (
    MutationSagaExecutor,
    build_journal,
    mutation_uow_context,
    uses_vector_index,
    validate_rollback_contract,
    write_lock_context,
)
from local_rag_backend.core.use_cases.docs_mutation_contracts import (
    MutationIntent,
    MutationUpsertInput,
    normalize_intent,
)

if TYPE_CHECKING:
    from local_rag_backend.core.ports.contracts import DocsMutationPorts
    from local_rag_backend.core.use_cases.results import MutationSummary
    from local_rag_backend.settings import Settings


def _new_op_id() -> str:
    return f"mut:{uuid.uuid4().hex}"


class MutationCoordinator:
    """Normalize/validate and execute each mutation under its own write lock."""

    def __init__(self, *, settings_obj: Settings, ports: DocsMutationPorts) -> None:
        self.settings_obj = settings_obj
        self.ports = ports
        self._saga = MutationSagaExecutor(settings_obj=settings_obj, ports=ports)

    def execute(self, intent: MutationIntent) -> MutationSummary:
        normalized = normalize_intent(intent=intent, new_op_id=_new_op_id)
        vector_mode_enabled = uses_vector_index(settings_obj=self.settings_obj)
        if vector_mode_enabled:
            validate_rollback_contract(ports=self.ports)

        journal = build_journal(ports=self.ports)
        if journal.get(normalized.op_id) is not None:
            with write_lock_context(settings_obj=self.settings_obj, ports=self.ports):
                replay = self._saga.get_committed_replay_summary(journal=journal, intent=normalized)
                if replay is not None:
                    return replay

        prepared = self._saga.prepare(
            intent=normalized,
            vector_mode_enabled=vector_mode_enabled,
        )

        with write_lock_context(settings_obj=self.settings_obj, ports=self.ports):
            return self._saga.execute_locked(prepared=prepared, journal=journal)

    def recover_incomplete(self, *, limit: int = 100) -> int:
        return self._saga.recover_incomplete(limit=limit)


__all__ = [
    "MutationCoordinator",
    "MutationIntent",
    "MutationSagaExecutor",
    "MutationUpsertInput",
    "build_journal",
    "mutation_uow_context",
    "normalize_intent",
    "uses_vector_index",
    "validate_rollback_contract",
    "write_lock_context",
]
