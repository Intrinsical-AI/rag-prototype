"""Runtime helpers for mutation coordinator orchestration concerns."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

from local_rag_backend.core.domain.profiles import StorageCapability
from local_rag_backend.core.ports.contracts import MutationRecord
from local_rag_backend.core.use_cases.docs_mutation_contracts import (
    MutationIntent,
    intent_to_dict,
    summary_to_payload,
)
from local_rag_backend.core.use_cases.results import MutationSummary

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from local_rag_backend.core.ports.contracts import DocsMutationPorts, MutationJournalPort
    from local_rag_backend.settings import Settings


def uses_vector_index(*, settings_obj: Settings) -> bool:
    return str(settings_obj.retrieval_mode) in ("dense", "hybrid")


def validate_storage_profile(
    *,
    settings_obj: Settings,
    ports: DocsMutationPorts,
    vector_mode_enabled: bool,
) -> None:
    profile = ports.storage_profile_registry.resolve(
        profile_id=getattr(settings_obj, "storage_profile", ""),
        retrieval_mode=settings_obj.retrieval_mode,
        vector_backend=getattr(settings_obj, "vector_backend", "auto"),
    )
    if profile.has(StorageCapability.READ_ONLY):
        raise RuntimeError(
            f"Storage profile {profile.profile_id!r} is read-only and cannot serve mutations."
        )
    if not profile.has(StorageCapability.DURABLE_SAGA):
        raise RuntimeError(
            f"Storage profile {profile.profile_id!r} does not satisfy DURABLE_SAGA writes."
        )
    if vector_mode_enabled and not profile.supports_vectors:
        raise RuntimeError(
            f"Storage profile {profile.profile_id!r} does not provide vector capabilities."
        )


def write_lock_context(*, settings_obj: Settings, ports: DocsMutationPorts) -> Any:
    timeout = float(getattr(settings_obj, "write_lock_timeout_s", 30.0))
    poll = float(getattr(settings_obj, "write_lock_poll_s", 0.05))
    return ports.write_lock(
        coordination_dir=settings_obj.get_coordination_dir(),
        timeout_s=timeout,
        poll_s=poll,
    )


def mutation_uow_context(*, ports: DocsMutationPorts) -> AbstractContextManager[None]:
    factory = ports.mutation_uow_factory
    if factory is None:
        return nullcontext()
    return factory()


def build_journal(*, ports: DocsMutationPorts) -> MutationJournalPort:
    return ports.mutation_journal_factory()


def upsert_journal_record(
    *,
    journal: MutationJournalPort,
    op_id: str,
    state: str,
    intent: MutationIntent,
    before_image: dict[str, Any] | None = None,
    outcome: MutationSummary | None = None,
    error: str | None = None,
) -> None:
    previous = journal.get(op_id)
    attempts = int(previous.attempts) + 1 if previous is not None else 1
    record = MutationRecord(
        op_id=op_id,
        state=cast("Any", state),
        intent=intent_to_dict(intent),
        before_image=before_image,
        outcome=summary_to_payload(outcome) if outcome is not None else None,
        error=error,
        attempts=attempts,
        created_at=previous.created_at if previous is not None else 0.0,
        updated_at=0.0,
    )
    journal.upsert(record)


__all__ = [
    "build_journal",
    "mutation_uow_context",
    "upsert_journal_record",
    "uses_vector_index",
    "validate_storage_profile",
    "write_lock_context",
]
