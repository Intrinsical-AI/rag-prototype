"""Application contracts (ports + result DTOs)."""

from local_rag_backend.app.contracts.ports import (
    DocsMutationPorts,
    DocsRepositoryPort,
    IndexMutationPorts,
    MutationJournalPort,
    MutationRecord,
    MutationState,
    UpsertDocBuilderPort,
    UpsertResultPort,
)
from local_rag_backend.app.contracts.results import (
    DeleteDocsByExternalIdSummary,
    DeleteDocsSummary,
    MutationSummary,
    UpsertDocResult,
    UpsertDocsSummary,
)

__all__ = [
    "DeleteDocsByExternalIdSummary",
    "DeleteDocsSummary",
    "DocsMutationPorts",
    "DocsRepositoryPort",
    "IndexMutationPorts",
    "MutationJournalPort",
    "MutationRecord",
    "MutationState",
    "MutationSummary",
    "UpsertDocBuilderPort",
    "UpsertDocResult",
    "UpsertDocsSummary",
    "UpsertResultPort",
]
