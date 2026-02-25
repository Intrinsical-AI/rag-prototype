"""Unified durable mutation coordinator for docs/index write operations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from local_rag_backend.core.domain.profiles import StorageCapability
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.ports.contracts import MutationRecord
from local_rag_backend.core.use_cases.results import MutationSummary, UpsertDocResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from local_rag_backend.core.ports.contracts import DocsMutationPorts, MutationJournalPort
    from local_rag_backend.settings import Settings


@dataclass(frozen=True)
class MutationUpsertInput:
    external_id: str
    content: str
    source_id: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MutationIntent:
    op_id: str
    upserts: tuple[MutationUpsertInput, ...] = ()
    delete_ids: tuple[str, ...] = ()
    delete_external_ids: tuple[str, ...] = ()
    source: str = "unknown"


@dataclass(frozen=True)
class _SqlMutationOutcome:
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


def _new_op_id() -> str:
    return f"mut:{uuid.uuid4().hex}"


def _normalize_str_items(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _summary_from_record(record: MutationRecord) -> MutationSummary:
    payload = dict(record.outcome or {})
    return MutationSummary(
        op_id=str(payload.get("op_id") or record.op_id),
        inserted=int(payload.get("inserted") or 0),
        updated=int(payload.get("updated") or 0),
        unchanged=int(payload.get("unchanged") or 0),
        deleted_sql=int(payload.get("deleted_sql") or 0),
        deleted_index=(
            int(payload["deleted_index"]) if payload.get("deleted_index") is not None else None
        ),
        tombstoned=int(payload.get("tombstoned") or 0),
        missing_external_ids=list(payload.get("missing_external_ids") or []),
        index_rebuilt=bool(payload.get("index_rebuilt") or False),
        index_doc_count=(
            int(payload["index_doc_count"]) if payload.get("index_doc_count") is not None else None
        ),
        results=[
            UpsertDocResult(
                external_id=str(item.get("external_id") or ""),
                id=str(item.get("id") or ""),
                action=str(item.get("action") or "unchanged"),
                content_changed=bool(item.get("content_changed") or False),
            )
            for item in list(payload.get("results") or [])
            if isinstance(item, dict)
        ],
    )


def _summary_to_payload(summary: MutationSummary) -> dict[str, Any]:
    return {
        "op_id": summary.op_id,
        "inserted": summary.inserted,
        "updated": summary.updated,
        "unchanged": summary.unchanged,
        "deleted_sql": summary.deleted_sql,
        "deleted_index": summary.deleted_index,
        "tombstoned": summary.tombstoned,
        "missing_external_ids": list(summary.missing_external_ids or []),
        "index_rebuilt": bool(summary.index_rebuilt),
        "index_doc_count": summary.index_doc_count,
        "results": [
            {
                "external_id": r.external_id,
                "id": r.id,
                "action": r.action,
                "content_changed": bool(r.content_changed),
            }
            for r in list(summary.results or [])
        ],
    }


class MutationCoordinator:
    def __init__(self, *, settings_obj: Settings, ports: DocsMutationPorts) -> None:
        self.settings_obj = settings_obj
        self.ports = ports

    def execute(self, intent: MutationIntent) -> MutationSummary:
        normalized = self._normalize_intent(intent)
        self._validate_storage_profile()
        journal = self._journal()

        existing = journal.get(normalized.op_id)
        if existing is not None:
            if existing.state == "COMMITTED":
                return _summary_from_record(existing)
            self._recover_record(journal=journal, record=existing)
            existing = journal.get(normalized.op_id)
            if existing is not None and existing.state == "COMMITTED":
                return _summary_from_record(existing)

        lock_root = self.settings_obj.get_coordination_dir()
        with self._write_lock(lock_root):
            existing_locked = journal.get(normalized.op_id)
            if existing_locked is not None and existing_locked.state == "COMMITTED":
                return _summary_from_record(existing_locked)

            doc_repo = self.ports.doc_repo_factory()
            before_image = self._capture_before_image(doc_repo=doc_repo, intent=normalized)
            self._upsert_record(
                journal=journal,
                op_id=normalized.op_id,
                state="PREPARED",
                intent=normalized,
                before_image=before_image,
            )

            sql_outcome = self._apply_sql_mutation(doc_repo=doc_repo, intent=normalized)
            self._upsert_record(
                journal=journal,
                op_id=normalized.op_id,
                state="SQL_COMMITTED",
                intent=normalized,
                before_image=before_image,
            )

            deleted_index: int | None = None
            index_doc_count: int | None = None
            try:
                if self._uses_vector_index():
                    deleted_index, index_doc_count = self._apply_vector_delta(
                        sql_outcome=sql_outcome,
                        doc_repo=doc_repo,
                    )
                    self._upsert_record(
                        journal=journal,
                        op_id=normalized.op_id,
                        state="VECTOR_COMMITTED",
                        intent=normalized,
                        before_image=before_image,
                    )
            except Exception as vector_err:
                self._upsert_record(
                    journal=journal,
                    op_id=normalized.op_id,
                    state="COMPENSATING",
                    intent=normalized,
                    before_image=before_image,
                    error=str(vector_err),
                )
                try:
                    self._rollback_sql(
                        doc_repo=doc_repo,
                        intent=self._intent_to_dict(normalized),
                        before_image=before_image,
                    )
                    self._upsert_record(
                        journal=journal,
                        op_id=normalized.op_id,
                        state="ROLLED_BACK",
                        intent=normalized,
                        before_image=before_image,
                        error=str(vector_err),
                    )
                except Exception as rollback_err:
                    self._upsert_record(
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
            self._upsert_record(
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
        journal = self._journal()
        records = journal.list_incomplete(limit=limit)
        if not records:
            return 0
        repaired = 0
        lock_root = self.settings_obj.get_coordination_dir()
        with self._write_lock(lock_root):
            for record in records:
                self._recover_record(journal=journal, record=record)
                repaired += 1
        return repaired

    def _write_lock(self, lock_root: Any) -> Any:
        timeout = float(getattr(self.settings_obj, "write_lock_timeout_s", 30.0))
        poll = float(getattr(self.settings_obj, "write_lock_poll_s", 0.05))
        return self.ports.write_lock(coordination_dir=lock_root, timeout_s=timeout, poll_s=poll)

    def _recover_record(self, *, journal: MutationJournalPort, record: MutationRecord) -> None:
        if record.state in {"COMMITTED", "ROLLED_BACK"}:
            return
        if record.state == "VECTOR_COMMITTED":
            committed = MutationRecord(
                op_id=record.op_id,
                state="COMMITTED",
                intent=dict(record.intent),
                before_image=record.before_image,
                outcome=record.outcome,
                error=record.error,
                attempts=int(record.attempts) + 1,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            journal.upsert(committed)
            journal.delete(record.op_id)
            return

        doc_repo = self.ports.doc_repo_factory()
        if record.before_image is None:
            rolled_back = MutationRecord(
                op_id=record.op_id,
                state="ROLLED_BACK",
                intent=dict(record.intent),
                before_image=None,
                outcome=record.outcome,
                error=record.error
                or "No before_image was available; record marked as rolled back.",
                attempts=int(record.attempts) + 1,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            journal.upsert(rolled_back)
            return

        compensating = MutationRecord(
            op_id=record.op_id,
            state="COMPENSATING",
            intent=dict(record.intent),
            before_image=record.before_image,
            outcome=record.outcome,
            error=record.error,
            attempts=int(record.attempts) + 1,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        journal.upsert(compensating)
        try:
            self._rollback_sql(
                doc_repo=doc_repo, intent=record.intent, before_image=record.before_image
            )
            rolled_back = MutationRecord(
                op_id=record.op_id,
                state="ROLLED_BACK",
                intent=dict(record.intent),
                before_image=record.before_image,
                outcome=record.outcome,
                error=record.error,
                attempts=int(record.attempts) + 1,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            journal.upsert(rolled_back)
        except Exception as rollback_err:
            failed = MutationRecord(
                op_id=record.op_id,
                state="FAILED_NEEDS_RECOVERY",
                intent=dict(record.intent),
                before_image=record.before_image,
                outcome=record.outcome,
                error=str(rollback_err),
                attempts=int(record.attempts) + 1,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            journal.upsert(failed)

    def _validate_storage_profile(self) -> None:
        registry = self.ports.storage_profile_registry
        profile = registry.resolve(
            profile_id=getattr(self.settings_obj, "storage_profile", ""),
            retrieval_mode=self.settings_obj.retrieval_mode,
            vector_backend=getattr(self.settings_obj, "vector_backend", "auto"),
        )
        if profile.has(StorageCapability.READ_ONLY):
            raise RuntimeError(
                f"Storage profile {profile.profile_id!r} is read-only and cannot serve mutations."
            )
        if not profile.has(StorageCapability.DURABLE_SAGA):
            raise RuntimeError(
                f"Storage profile {profile.profile_id!r} does not satisfy DURABLE_SAGA writes."
            )
        if self._uses_vector_index() and not profile.supports_vectors:
            raise RuntimeError(
                f"Storage profile {profile.profile_id!r} does not provide vector capabilities."
            )

    def _journal(self) -> MutationJournalPort:
        return self.ports.mutation_journal_factory()

    def _uses_vector_index(self) -> bool:
        return str(self.settings_obj.retrieval_mode) in ("dense", "hybrid")

    def _normalize_intent(self, intent: MutationIntent) -> MutationIntent:
        upserts = [
            MutationUpsertInput(
                external_id=str(it.external_id).strip(),
                content=str(it.content).strip(),
                source_id=(str(it.source_id) if it.source_id is not None else None),
                metadata=dict(it.metadata) if it.metadata is not None else None,
            )
            for it in list(intent.upserts)
            if str(it.external_id).strip() and str(it.content).strip()
        ]
        ext_ids = _normalize_str_items(intent.delete_external_ids)
        delete_ids = _normalize_str_items(intent.delete_ids)

        if not upserts and not ext_ids and not delete_ids:
            raise ValueError("Mutation intent must include upserts and/or deletions.")

        upsert_ext_ids = [it.external_id for it in upserts]
        if len(set(upsert_ext_ids)) != len(upsert_ext_ids):
            raise ValueError("external_id values in upserts must be unique per mutation intent.")
        ext_conflict = sorted(set(upsert_ext_ids) & set(ext_ids))
        if ext_conflict:
            raise ValueError(
                "upserts and delete_external_ids cannot target the same external_id values: "
                + ", ".join(ext_conflict[:10])
            )

        return MutationIntent(
            op_id=str(intent.op_id).strip() or _new_op_id(),
            upserts=tuple(upserts),
            delete_ids=tuple(delete_ids),
            delete_external_ids=tuple(ext_ids),
            source=str(intent.source or "unknown"),
        )

    def _capture_before_image(self, *, doc_repo: Any, intent: MutationIntent) -> dict[str, Any]:
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

        # Deduplicate snapshots by doc_id (later captures overwrite earlier duplicates).
        by_doc_id = {str(s.get("id")): s for s in snapshots if str(s.get("id") or "").strip()}
        return {
            "docs": list(by_doc_id.values()),
            "existing_tombstones": existing_tombstones,
        }

    def _apply_sql_mutation(self, *, doc_repo: Any, intent: MutationIntent) -> _SqlMutationOutcome:
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
            items = [
                self.ports.build_upsert_doc(
                    external_id=u.external_id,
                    content=u.content,
                    source_id=u.source_id,
                    metadata=u.metadata,
                )
                for u in intent.upserts
            ]
            upsert_results, changed, updated_ids = doc_repo.upsert_documents_by_external_id(items)
            changed_content.extend(
                [(DocId(str(doc_id)), str(content)) for doc_id, content in changed]
            )
            updated_content_ids.extend([DocId(str(doc_id)) for doc_id in updated_ids])
            for r in upsert_results:
                action = str(r.action)
                if action == "inserted":
                    inserted += 1
                elif action == "updated":
                    updated += 1
                else:
                    unchanged += 1
                results.append(
                    UpsertDocResult(
                        external_id=str(r.external_id),
                        id=str(r.id),
                        action=action,
                        content_changed=bool(r.content_changed),
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
        return _SqlMutationOutcome(
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
        self,
        *,
        sql_outcome: _SqlMutationOutcome,
        doc_repo: Any,
    ) -> tuple[int | None, int | None]:
        if not self._uses_vector_index():
            return None, None

        delete_ids = [
            DocId(str(x))
            for x in {
                *[str(doc_id) for doc_id in sql_outcome.updated_content_ids],
                *[str(doc_id) for doc_id in sql_outcome.deleted_doc_ids],
            }
        ]

        needs_upsert_vectors = bool(sql_outcome.changed_content)
        embedder = self.ports.build_embedder() if needs_upsert_vectors else None
        dim = int(embedder.dim) if embedder is not None else None
        vec_repo = self.ports.vector_repo_factory(
            index_path=self.settings_obj.index_path,
            id_map_path=self.settings_obj.id_map_path,
            dim=dim,
            backend=getattr(self.settings_obj, "vector_backend", "auto"),
        )

        upsert_vectors: list[tuple[DocId, list[float]]] = []
        if embedder is not None:
            embed_inputs = [content for _, content in sql_outcome.changed_content]
            vectors = embedder.embed(embed_inputs)
            if len(vectors) != len(embed_inputs):
                raise RuntimeError(
                    f"Embedder returned {len(vectors)} vectors for {len(embed_inputs)} changed documents."
                )
            upsert_vectors = [
                (doc_id, list(vec))
                for (doc_id, _content), vec in zip(
                    sql_outcome.changed_content, vectors, strict=False
                )
            ]

        apply_delta = getattr(vec_repo, "apply_delta_atomic", None)
        if not callable(apply_delta):
            raise RuntimeError(
                "Vector adapter must implement apply_delta_atomic for DURABLE_SAGA mutations."
            )
        apply_delta(delete_ids=delete_ids, upserts=upsert_vectors)

        index_doc_count = (
            len(list(doc_repo.get_all_documents()))
            if hasattr(doc_repo, "get_all_documents")
            else None
        )
        return len(delete_ids), index_doc_count

    def _rollback_sql(
        self, *, doc_repo: Any, intent: dict[str, Any], before_image: dict[str, Any]
    ) -> None:
        upserts = [
            str(x.get("external_id") or "").strip() for x in list(intent.get("upserts") or [])
        ]
        delete_external_ids = _normalize_str_items(list(intent.get("delete_external_ids") or []))
        affected_external_ids = _normalize_str_items([*upserts, *delete_external_ids])

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

    def _upsert_record(
        self,
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
            intent=self._intent_to_dict(intent),
            before_image=before_image,
            outcome=_summary_to_payload(outcome) if outcome is not None else None,
            error=error,
            attempts=attempts,
            created_at=previous.created_at if previous is not None else 0.0,
            updated_at=0.0,
        )
        journal.upsert(record)

    def _intent_to_dict(self, intent: MutationIntent) -> dict[str, Any]:
        return {
            "op_id": intent.op_id,
            "source": intent.source,
            "upserts": [
                {
                    "external_id": u.external_id,
                    "content": u.content,
                    "source_id": u.source_id,
                    "metadata": dict(u.metadata) if u.metadata is not None else None,
                }
                for u in intent.upserts
            ],
            "delete_ids": list(intent.delete_ids),
            "delete_external_ids": list(intent.delete_external_ids),
        }


__all__ = ["MutationCoordinator", "MutationIntent", "MutationUpsertInput"]
