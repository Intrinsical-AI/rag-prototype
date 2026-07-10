"""Unified durable mutation coordinator for docs/index write operations."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, cast

from local_rag_backend.core.domain.profiles import StorageCapability
from local_rag_backend.core.use_cases._atomic_mutation_executor import AtomicMutationExecutor
from local_rag_backend.core.use_cases._batch_coordinator import (
    MutationBatchCoordinator,
    MutationBatchItem,
)
from local_rag_backend.core.use_cases._mutation_saga_executor import (
    MutationSagaExecutor,
    PreparedMutation,
    build_journal,
    mutation_uow_context,
    uses_vector_index,
    validate_rollback_contract,
    validate_storage_profile,
    write_lock_context,
)
from local_rag_backend.core.use_cases.docs_mutation_contracts import (
    MutationIntent,
    MutationUpsertInput,
    normalize_intent,
)

if TYPE_CHECKING:
    from local_rag_backend.core.ports.contracts import DocsMutationPorts, MutationJournalPort
    from local_rag_backend.core.use_cases.results import MutationSummary
    from local_rag_backend.settings import Settings


def _new_op_id() -> str:
    return f"mut:{uuid.uuid4().hex}"


class MutationCoordinator:
    """Thin orchestration layer: normalize/validate, batch, and delegate saga execution."""

    def __init__(self, *, settings_obj: Settings, ports: DocsMutationPorts) -> None:
        self.settings_obj = settings_obj
        self.ports = ports
        self._batcher = MutationBatchCoordinator()
        self._saga = MutationSagaExecutor(settings_obj=settings_obj, ports=ports)
        self._atomic = AtomicMutationExecutor(settings_obj=settings_obj, ports=ports)

    def execute(self, intent: MutationIntent) -> MutationSummary:
        normalized = normalize_intent(intent=intent, new_op_id=_new_op_id)
        vector_mode_enabled = uses_vector_index(settings_obj=self.settings_obj)
        validate_storage_profile(
            settings_obj=self.settings_obj,
            ports=self.ports,
            vector_mode_enabled=vector_mode_enabled,
        )
        profile = self.ports.storage_profile_registry.resolve(
            profile_id=getattr(self.settings_obj, "storage_profile", ""),
            persistence_backend=getattr(self.settings_obj, "persistence_backend", "local_split"),
            retrieval_mode=self.settings_obj.retrieval_mode,
            vector_backend=getattr(self.settings_obj, "vector_backend", "auto"),
        )
        use_atomic = profile.has(StorageCapability.ATOMIC) and not profile.has(
            StorageCapability.DURABLE_SAGA
        )
        if vector_mode_enabled and not use_atomic:
            validate_rollback_contract(ports=self.ports)

        journal = None
        if not use_atomic:
            journal = build_journal(ports=self.ports)
            replay = self._saga.get_committed_replay_summary(journal=journal, intent=normalized)
            if replay is not None:
                return replay

        precomputed_vectors = self._saga.precompute_vectors_for_intent(
            intent=normalized,
            vector_mode_enabled=vector_mode_enabled,
        )
        prepared = PreparedMutation(
            intent=normalized,
            vector_mode_enabled=vector_mode_enabled,
            precomputed_vectors_by_external_id=precomputed_vectors,
        )

        return cast(
            "MutationSummary",
            self._batcher.submit(
                queue_key=self._batch_state_key(),
                payload=prepared,
                max_batch_size=self._batch_max_size(),
                max_wait_ms=self._batch_max_wait_ms(),
                process_batch=lambda batch: self._process_batch(
                    batch=batch,
                    journal=journal,
                    use_atomic=use_atomic,
                ),
            ),
        )

    def recover_incomplete(self, *, limit: int = 100) -> int:
        profile = self.ports.storage_profile_registry.resolve(
            profile_id=getattr(self.settings_obj, "storage_profile", ""),
            persistence_backend=getattr(self.settings_obj, "persistence_backend", "local_split"),
            retrieval_mode=self.settings_obj.retrieval_mode,
            vector_backend=getattr(self.settings_obj, "vector_backend", "auto"),
        )
        if profile.has(StorageCapability.ATOMIC) and not profile.has(
            StorageCapability.DURABLE_SAGA
        ):
            return self._atomic.recover_incomplete(limit=limit)
        return self._saga.recover_incomplete(limit=limit)

    def _batch_state_key(self) -> str:
        coordination_dir = Path(self.settings_obj.get_coordination_dir())
        return str(coordination_dir.resolve())

    def _batch_max_size(self) -> int:
        raw = int(getattr(self.settings_obj, "mutation_batch_max_size", 32))
        return max(1, min(raw, 512))

    def _batch_max_wait_ms(self) -> int:
        raw = int(getattr(self.settings_obj, "mutation_batch_max_wait_ms", 50))
        return max(0, min(raw, 5000))

    def _process_batch(
        self,
        *,
        batch: list[MutationBatchItem],
        journal: MutationJournalPort | None,
        use_atomic: bool,
    ) -> None:
        with write_lock_context(settings_obj=self.settings_obj, ports=self.ports):
            for item in batch:
                try:
                    if use_atomic:
                        item.result = self._atomic.execute_locked(
                            prepared=cast("PreparedMutation", item.payload)
                        )
                    else:
                        if journal is None:  # pragma: no cover
                            raise RuntimeError(
                                "Mutation journal is required for durable saga mode."
                            )
                        item.result = self._saga.execute_locked(
                            prepared=cast("PreparedMutation", item.payload),
                            journal=journal,
                        )
                except Exception as exc:
                    item.error = exc
                finally:
                    item.done.set()


__all__ = [
    "MutationBatchCoordinator",
    "MutationCoordinator",
    "MutationIntent",
    "MutationSagaExecutor",
    "MutationUpsertInput",
    "build_journal",
    "mutation_uow_context",
    "normalize_intent",
    "uses_vector_index",
    "validate_rollback_contract",
    "validate_storage_profile",
    "write_lock_context",
]
