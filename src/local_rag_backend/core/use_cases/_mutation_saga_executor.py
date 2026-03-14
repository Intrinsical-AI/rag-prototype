"""Durable saga executor for document mutation flows."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from local_rag_backend.core.domain.profiles import StorageCapability
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.ports.contracts import MutationRecord
from local_rag_backend.core.services.dense_upsert import precompute_vectors_for_changed_items
from local_rag_backend.core.use_cases.docs_mutation_contracts import (
    MutationIntent,
    canonical_intent_payload,
    intent_to_dict,
    normalize_str_items,
    summary_to_payload,
)
from local_rag_backend.core.use_cases.results import MutationSummary, UpsertDocResult

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from local_rag_backend.core.ports.contracts import (
        DocsMutationPorts,
        MutationJournalPort,
        MutationState,
        UpsertDocBuilderPort,
    )
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
        persistence_backend=getattr(settings_obj, "persistence_backend", "local_split"),
        retrieval_mode=settings_obj.retrieval_mode,
        vector_backend=getattr(settings_obj, "vector_backend", "auto"),
    )
    if profile.has(StorageCapability.READ_ONLY):
        raise RuntimeError(
            f"Storage profile {profile.profile_id!r} is read-only and cannot serve mutations."
        )
    if not profile.has(StorageCapability.DURABLE_SAGA) and not profile.has(StorageCapability.ATOMIC):
        raise RuntimeError(
            f"Storage profile {profile.profile_id!r} does not satisfy writable storage capabilities."
        )
    if vector_mode_enabled and not profile.supports_vectors:
        raise RuntimeError(
            f"Storage profile {profile.profile_id!r} does not provide vector capabilities."
        )


def validate_rollback_contract(*, ports: DocsMutationPorts) -> None:
    doc_repo = ports.doc_repo_factory()
    required_methods = (
        "snapshot_by_external_ids",
        "hard_delete_by_external_ids",
        "restore_from_snapshots",
    )
    missing = [name for name in required_methods if not callable(getattr(doc_repo, name, None))]
    if missing:
        raise RuntimeError(
            "DURABLE_SAGA rollback contract missing required docs adapter methods: "
            + ", ".join(missing)
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


@dataclass(frozen=True)
class SqlMutationOutcome:
    inserted: int
    updated: int
    unchanged: int
    results: list[UpsertDocResult]
    changed_content: list[tuple[DocId, str]]
    updated_content_ids: list[DocId]
    deleted_sql: int
    deleted_doc_ids: list[DocId]
    tombstoned: int
    missing_external_ids: list[str]


@dataclass(frozen=True)
class PreparedMutation:
    intent: MutationIntent
    vector_mode_enabled: bool
    precomputed_vectors_by_external_id: dict[str, list[float]]


class MutationSagaExecutor:
    def __init__(self, *, settings_obj: Settings, ports: DocsMutationPorts) -> None:
        self.settings_obj = settings_obj
        self.ports = ports

    def get_committed_replay_summary(
        self,
        *,
        journal: MutationJournalPort,
        intent: MutationIntent,
    ) -> MutationSummary | None:
        existing = journal.get(intent.op_id)
        if existing is None:
            return None
        self._assert_replay_compatible(existing_intent=existing.intent, intent=intent)
        return None

    def precompute_vectors_for_intent(
        self,
        *,
        intent: MutationIntent,
        vector_mode_enabled: bool,
    ) -> dict[str, list[float]]:
        if not vector_mode_enabled or not intent.upserts:
            return {}

        embedder = self.ports.build_embedder()
        doc_repo = self.ports.doc_repo_factory()

        if hasattr(doc_repo, "get_existing_doc_states_by_external_id"):
            return precompute_vectors_for_changed_items(
                items=intent.upserts,
                doc_repo=cast("Any", doc_repo),
                embedder=embedder,
            )

        inputs = [item.content.strip() for item in intent.upserts]
        vectors = embedder.embed(inputs)
        if len(vectors) != len(inputs):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(inputs)} documents."
            )
        return {
            item.external_id: list(vec) for item, vec in zip(intent.upserts, vectors, strict=False)
        }

    def execute_locked(
        self,
        *,
        prepared: PreparedMutation,
        journal: MutationJournalPort,
    ) -> MutationSummary:
        normalized = prepared.intent

        existing = journal.get(normalized.op_id)
        if existing is not None:
            self._assert_replay_compatible(existing_intent=existing.intent, intent=normalized)
            if existing.state != "COMMITTED":
                _recover_record(
                    journal=journal,
                    record=existing,
                    rollback_sql_fn=self._rollback_sql_with_uow,
                    doc_repo_factory=self.ports.doc_repo_factory,
                )
                existing = journal.get(normalized.op_id)
                if existing is not None and existing.state == "COMMITTED":
                    existing = None

        doc_repo = self.ports.doc_repo_factory()
        before_image = _capture_before_image(doc_repo=doc_repo, intent=normalized)
        _upsert_journal_record(
            journal=journal,
            op_id=normalized.op_id,
            state="PREPARED",
            intent=normalized,
            before_image=before_image,
        )

        with mutation_uow_context(ports=self.ports):
            sql_outcome = _apply_sql_mutation(
                doc_repo=doc_repo,
                intent=normalized,
                build_upsert_doc=self.ports.build_upsert_doc,
            )
        _upsert_journal_record(
            journal=journal,
            op_id=normalized.op_id,
            state="SQL_COMMITTED",
            intent=normalized,
            before_image=before_image,
        )

        deleted_index: int | None = None
        index_doc_count: int | None = None
        try:
            if prepared.vector_mode_enabled:
                deleted_index, index_doc_count = _apply_vector_delta(
                    sql_outcome=sql_outcome,
                    doc_repo=doc_repo,
                    settings_obj=self.settings_obj,
                    ports=self.ports,
                    vector_mode_enabled=prepared.vector_mode_enabled,
                    vectors_by_external_id=prepared.precomputed_vectors_by_external_id,
                )
                _upsert_journal_record(
                    journal=journal,
                    op_id=normalized.op_id,
                    state="VECTOR_COMMITTED",
                    intent=normalized,
                    before_image=before_image,
                )
        except Exception as vector_err:
            _upsert_journal_record(
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
                _upsert_journal_record(
                    journal=journal,
                    op_id=normalized.op_id,
                    state="ROLLED_BACK",
                    intent=normalized,
                    before_image=before_image,
                    error=str(vector_err),
                )
            except Exception as rollback_err:
                _upsert_journal_record(
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
        _upsert_journal_record(
            journal=journal,
            op_id=normalized.op_id,
            state="COMMITTED",
            intent=normalized,
            before_image=before_image,
            outcome=summary,
        )
        return summary

    def recover_incomplete(self, *, limit: int = 100) -> int:
        journal = build_journal(ports=self.ports)
        records = journal.list_incomplete(limit=limit)
        if not records:
            return 0

        repaired = 0
        with write_lock_context(settings_obj=self.settings_obj, ports=self.ports):
            for record in records:
                _recover_record(
                    journal=journal,
                    record=record,
                    rollback_sql_fn=self._rollback_sql_with_uow,
                    doc_repo_factory=self.ports.doc_repo_factory,
                )
                repaired += 1
            if repaired:
                self._reconcile_vector_from_sql()
        return repaired

    def _reconcile_vector_from_sql(self) -> None:
        if not uses_vector_index(settings_obj=self.settings_obj):
            return

        embedder = self.ports.build_embedder()
        vec_repo = self.ports.vector_repo_factory(
            index_path=self.settings_obj.index_path,
            id_map_path=self.settings_obj.id_map_path,
            dim=int(embedder.dim),
            backend=getattr(self.settings_obj, "vector_backend", "auto"),
        )
        doc_repo = self.ports.doc_repo_factory()
        self.ports.rebuild_fn(doc_repo=doc_repo, vec_repo=vec_repo, embedder=embedder)

    def _rollback_sql_with_uow(
        self,
        *,
        doc_repo: Any,
        intent: dict[str, Any],
        before_image: dict[str, Any],
    ) -> None:
        with mutation_uow_context(ports=self.ports):
            _rollback_sql(doc_repo=doc_repo, intent=intent, before_image=before_image)

    def _assert_replay_compatible(
        self, *, existing_intent: dict[str, Any], intent: MutationIntent
    ) -> None:
        if canonical_intent_payload(existing_intent) != canonical_intent_payload(
            intent_to_dict(intent)
        ):
            raise ValueError(
                "Mutation op_id replay mismatch: this op_id already exists with a different intent."
            )


def _upsert_journal_record(
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


def _capture_before_image(*, doc_repo: Any, intent: MutationIntent) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []
    if intent.upserts or intent.delete_external_ids:
        external_ids = sorted(
            {
                *(u.external_id for u in intent.upserts),
                *intent.delete_external_ids,
            }
        )
        if hasattr(doc_repo, "snapshot_by_external_ids"):
            ext_snaps = list(doc_repo.snapshot_by_external_ids(external_ids))
            snapshots.extend([dict(cast("dict[str, Any]", s)) for s in ext_snaps])

    if intent.delete_ids and hasattr(doc_repo, "snapshot_by_ids"):
        id_snaps = list(doc_repo.snapshot_by_ids([DocId(str(x)) for x in intent.delete_ids]))
        snapshots.extend([dict(cast("dict[str, Any]", s)) for s in id_snaps])

    existing_tombstones: list[str] = []
    if intent.delete_external_ids and hasattr(doc_repo, "get_tombstoned_external_ids"):
        existing_tombstones = sorted(
            doc_repo.get_tombstoned_external_ids(list(intent.delete_external_ids))
        )

    by_doc_id = {str(s.get("id")): s for s in snapshots if str(s.get("id") or "").strip()}
    return {
        "docs": list(by_doc_id.values()),
        "existing_tombstones": existing_tombstones,
    }


def _apply_sql_mutation(
    *,
    doc_repo: Any,
    intent: MutationIntent,
    build_upsert_doc: UpsertDocBuilderPort,
) -> SqlMutationOutcome:
    inserted = 0
    updated = 0
    unchanged = 0
    results: list[UpsertDocResult] = []
    changed_content: list[tuple[DocId, str]] = []
    updated_content_ids: list[DocId] = []
    deleted_sql = 0
    deleted_doc_ids: list[DocId] = []
    tombstoned = 0
    missing_external_ids: list[str] = []

    if intent.upserts:
        upsert_external_ids = [str(u.external_id) for u in intent.upserts]
        if hasattr(doc_repo, "get_tombstoned_external_ids"):
            tombstoned_external_ids = sorted(
                doc_repo.get_tombstoned_external_ids(upsert_external_ids)
            )
            if tombstoned_external_ids:
                blocked = ", ".join(tombstoned_external_ids[:10])
                raise ValueError(
                    "Cannot upsert tombstoned external_id values: "
                    + blocked
                    + (", ..." if len(tombstoned_external_ids) > 10 else "")
                )

        items = [
            build_upsert_doc(
                external_id=u.external_id,
                content=u.content,
                source_id=u.source_id,
                scope=u.scope,
                snapshot_id=u.snapshot_id,
                metadata=u.metadata,
            )
            for u in intent.upserts
        ]
        upsert_results, changed, updated_ids = doc_repo.upsert_documents_by_external_id(items)
        changed_content.extend([(DocId(str(doc_id)), str(content)) for doc_id, content in changed])
        updated_content_ids.extend([DocId(str(doc_id)) for doc_id in updated_ids])

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
        deleted_doc_ids.extend([DocId(str(x)) for x in list(deleted_ids)])
        missing_external_ids.extend([str(x) for x in list(missing)])
        tombstoned += int(tombstoned_count)

    dedup_deleted_ids = list({str(x): DocId(str(x)) for x in deleted_doc_ids}.values())
    return SqlMutationOutcome(
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        results=results,
        changed_content=changed_content,
        updated_content_ids=updated_content_ids,
        deleted_sql=deleted_sql,
        deleted_doc_ids=dedup_deleted_ids,
        tombstoned=tombstoned,
        missing_external_ids=missing_external_ids,
    )


def _apply_vector_delta(
    *,
    sql_outcome: SqlMutationOutcome,
    doc_repo: Any,
    settings_obj: Settings,
    ports: DocsMutationPorts,
    vector_mode_enabled: bool,
    vectors_by_external_id: dict[str, list[float]] | None = None,
) -> tuple[int | None, int | None]:
    if not vector_mode_enabled:
        return None, None

    delete_ids = [
        DocId(str(x))
        for x in {
            *[str(doc_id) for doc_id in sql_outcome.updated_content_ids],
            *[str(doc_id) for doc_id in sql_outcome.deleted_doc_ids],
        }
    ]

    dim: int | None = None
    if vectors_by_external_id:
        first_vec = next(iter(vectors_by_external_id.values()), None)
        if first_vec is not None:
            dim = len(first_vec)

    vec_repo = ports.vector_repo_factory(
        index_path=settings_obj.index_path,
        id_map_path=settings_obj.id_map_path,
        dim=dim,
        backend=getattr(settings_obj, "vector_backend", "auto"),
    )

    upsert_vectors: list[tuple[DocId, list[float]]] = []
    if sql_outcome.changed_content:
        precomputed = vectors_by_external_id or {}
        if not precomputed:
            raise RuntimeError(
                "Missing precomputed vectors for changed documents in durable mutation flow."
            )

        external_id_by_doc_id = {
            str(r.id): r.external_id for r in sql_outcome.results if bool(r.content_changed)
        }
        missing: list[str] = []
        for doc_id, _content in sql_outcome.changed_content:
            ext_id = external_id_by_doc_id.get(str(doc_id))
            if not ext_id:
                missing.append(str(doc_id))
                continue
            vector = precomputed.get(ext_id)
            if vector is None:
                missing.append(str(ext_id))
                continue
            upsert_vectors.append((doc_id, list(vector)))

        if missing:
            raise RuntimeError(
                "Missing precomputed vectors for changed documents: " + ", ".join(missing[:10])
            )

    apply_delta = getattr(vec_repo, "apply_delta_atomic", None)
    if not callable(apply_delta):
        raise RuntimeError(
            "Vector adapter must implement apply_delta_atomic for DURABLE_SAGA mutations."
        )
    apply_delta(delete_ids=delete_ids, upserts=upsert_vectors)

    index_doc_count = (
        len(list(doc_repo.get_all_documents())) if hasattr(doc_repo, "get_all_documents") else None
    )
    return len(delete_ids), index_doc_count


def _rollback_sql(*, doc_repo: Any, intent: dict[str, Any], before_image: dict[str, Any]) -> None:
    upserts = [str(x.get("external_id") or "").strip() for x in list(intent.get("upserts") or [])]
    delete_external_ids = normalize_str_items(list(intent.get("delete_external_ids") or []))
    affected_external_ids = normalize_str_items([*upserts, *delete_external_ids])

    if affected_external_ids and hasattr(doc_repo, "hard_delete_by_external_ids"):
        doc_repo.hard_delete_by_external_ids(affected_external_ids)

    snapshots = list(before_image.get("docs") or [])
    if snapshots and hasattr(doc_repo, "restore_from_snapshots"):
        doc_repo.restore_from_snapshots(snapshots)

    if delete_external_ids and hasattr(doc_repo, "delete_tombstones"):
        existing_before = set(before_image.get("existing_tombstones") or [])
        created_now = [ext for ext in delete_external_ids if ext not in existing_before]
        if created_now:
            doc_repo.delete_tombstones(created_now)


def _recover_record(
    *,
    journal: MutationJournalPort,
    record: MutationRecord,
    rollback_sql_fn: Any,
    doc_repo_factory: Any,
) -> None:
    if record.state in {"COMMITTED", "ROLLED_BACK"}:
        return
    if record.state == "VECTOR_COMMITTED":
        committed = _clone_record(record=record, state="COMMITTED")
        journal.upsert(committed)
        journal.delete(record.op_id)
        return

    doc_repo = doc_repo_factory()
    if record.before_image is None:
        rolled_back = _clone_record(
            record=record,
            state="ROLLED_BACK",
            before_image=None,
            error=record.error or "No before_image was available; record marked as rolled back.",
        )
        journal.upsert(rolled_back)
        return

    compensating = _clone_record(record=record, state="COMPENSATING")
    journal.upsert(compensating)
    try:
        rollback_sql_fn(doc_repo=doc_repo, intent=record.intent, before_image=record.before_image)
        rolled_back = _clone_record(record=record, state="ROLLED_BACK")
        journal.upsert(rolled_back)
    except Exception as rollback_err:
        failed = _clone_record(
            record=record, state="FAILED_NEEDS_RECOVERY", error=str(rollback_err)
        )
        journal.upsert(failed)


def _clone_record(
    *,
    record: MutationRecord,
    state: MutationState,
    before_image: dict[str, Any] | None | object = ...,
    error: str | None | object = ...,
) -> MutationRecord:
    return MutationRecord(
        op_id=record.op_id,
        state=state,
        intent=dict(record.intent),
        before_image=(
            record.before_image
            if before_image is ...
            else cast("dict[str, Any] | None", before_image)
        ),
        outcome=record.outcome,
        error=record.error if error is ... else cast("str | None", error),
        attempts=int(record.attempts) + 1,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


__all__ = [
    "MutationSagaExecutor",
    "PreparedMutation",
    "build_journal",
    "mutation_uow_context",
    "uses_vector_index",
    "validate_rollback_contract",
    "validate_storage_profile",
    "write_lock_context",
]
