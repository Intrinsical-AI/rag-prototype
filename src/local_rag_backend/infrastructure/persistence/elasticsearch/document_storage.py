"""Unified Elasticsearch-backed document and vector storage."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from local_rag_backend.core.domain.entities import Document as DomainDocument
from local_rag_backend.core.domain.types import DocId, new_doc_id
from local_rag_backend.core.ports import DocumentRepoPort, VectorRepoPort
from local_rag_backend.infrastructure.persistence.elasticsearch.client import (
    ElasticBackendError,
    ElasticClient,
)
from local_rag_backend.settings import Settings, settings as global_settings


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class _SearchResult:
    id: str
    score: float


class ElasticDocsRepository(DocumentRepoPort):
    @dataclass(frozen=True)
    class UpsertDoc:
        external_id: str
        content: str
        source_id: str | None = None
        scope: str | None = None
        snapshot_id: str | None = None
        metadata: Mapping[str, Any] | None = None
        chunk_dedup_sha256: str | None = None
        embedding: Sequence[float] | None = None

    @dataclass(frozen=True)
    class UpsertResult:
        external_id: str
        id: DocId
        action: Literal["inserted", "updated", "unchanged"]
        content_changed: bool

    @dataclass(frozen=True)
    class ExistingDocState:
        id: DocId
        external_id: str
        content: str
        content_sha256: str | None
        source_id: str | None
        scope: str | None
        snapshot_id: str | None
        metadata: dict[str, Any] | None
        chunk_dedup_sha256: str | None

    def __init__(
        self,
        *,
        settings_obj: Settings | None = None,
        client: ElasticClient | None = None,
    ) -> None:
        self._settings = settings_obj or global_settings
        self._client = client or ElasticClient(settings_obj=self._settings)
        self._client.ensure_indices(embed_dim=None)

    def store_documents(self, texts: Sequence[str]) -> list[DocId]:
        items = [
            ElasticDocsRepository.UpsertDoc(
                external_id=str(new_doc_id(prefix="doc")),
                content=str(text),
            )
            for text in texts
            if str(text).strip()
        ]
        results, _changed, _updated = self.upsert_documents_by_external_id(items)
        return [DocId(str(result.id)) for result in results]

    def delete_documents(self, ids: Sequence[DocId]) -> None:
        ops = [
            {"delete": {"_index": str(self._settings.es_docs_index), "_id": str(doc_id)}}
            for doc_id in ids
            if str(doc_id).strip()
        ]
        if ops:
            self._client.bulk(ops)

    def get(self, ids: Sequence[DocId]) -> Sequence[DomainDocument]:
        docs = self._client.mget(index=str(self._settings.es_docs_index), ids=[str(x) for x in ids])
        docs_by_id = {
            str(doc.get("_id")): self._to_domain_document(doc)
            for doc in docs
            if bool(doc.get("found"))
        }
        return [docs_by_id[str(doc_id)] for doc_id in ids if str(doc_id) in docs_by_id]

    def get_all_documents(self) -> Sequence[DomainDocument]:
        return self._scan_documents()

    def get_existing_doc_states_by_external_id(
        self, external_ids: Sequence[str]
    ) -> dict[str, ExistingDocState]:
        docs = self._client.mget(
            index=str(self._settings.es_docs_index),
            ids=[str(x) for x in external_ids if str(x).strip()],
        )
        out: dict[str, ElasticDocsRepository.ExistingDocState] = {}
        for doc in docs:
            if not bool(doc.get("found")):
                continue
            source = dict(doc.get("_source") or {})
            ext_id = str(source.get("external_id") or doc.get("_id") or "")
            if not ext_id:
                continue
            out[ext_id] = ElasticDocsRepository.ExistingDocState(
                id=DocId(ext_id),
                external_id=ext_id,
                content=str(source.get(self._settings.es_content_field) or ""),
                content_sha256=(
                    str(source.get("content_sha256"))
                    if source.get("content_sha256") is not None
                    else None
                ),
                source_id=(
                    str(source.get("source_id")) if source.get("source_id") is not None else None
                ),
                scope=(str(source.get("scope")) if source.get("scope") is not None else None),
                snapshot_id=(
                    str(source.get("snapshot_id"))
                    if source.get("snapshot_id") is not None
                    else None
                ),
                metadata=dict(source.get("metadata") or {}),
                chunk_dedup_sha256=(
                    str(source.get("chunk_dedup_sha256"))
                    if source.get("chunk_dedup_sha256") is not None
                    else None
                ),
            )
        return out

    def upsert_documents_by_external_id(
        self, items: Sequence[UpsertDoc]
    ) -> tuple[list[UpsertResult], list[tuple[DocId, str]], list[DocId]]:
        items_list = [item for item in items if str(item.external_id).strip()]
        if not items_list:
            return [], [], []

        ext_ids = [str(item.external_id).strip() for item in items_list]
        if len(set(ext_ids)) != len(ext_ids):
            raise ValueError("external_id values must be unique within the request")

        existing = self.get_existing_doc_states_by_external_id(ext_ids)
        results: list[ElasticDocsRepository.UpsertResult] = []
        changed: list[tuple[DocId, str]] = []
        updated_ids: list[DocId] = []
        ops: list[dict[str, Any]] = []

        for item in items_list:
            external_id = str(item.external_id).strip()
            content = str(item.content)
            content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            existing_doc = existing.get(external_id)

            action: Literal["inserted", "updated", "unchanged"]
            content_changed = True
            if existing_doc is None:
                action = "inserted"
            else:
                content_changed = (
                    (existing_doc.content_sha256 or "") != content_sha
                    or existing_doc.content != content
                )
                metadata_changed = (
                    item.metadata is not None
                    and dict(item.metadata) != dict(existing_doc.metadata or {})
                )
                source_changed = (
                    item.source_id is not None and str(item.source_id) != existing_doc.source_id
                )
                scope_changed = item.scope is not None and str(item.scope) != existing_doc.scope
                snapshot_changed = (
                    item.snapshot_id is not None
                    and str(item.snapshot_id) != existing_doc.snapshot_id
                )
                dedup_changed = (
                    item.chunk_dedup_sha256 is not None
                    and str(item.chunk_dedup_sha256) != existing_doc.chunk_dedup_sha256
                )
                if (
                    content_changed
                    or metadata_changed
                    or source_changed
                    or scope_changed
                    or snapshot_changed
                    or dedup_changed
                ):
                    action = "updated"
                else:
                    action = "unchanged"

            if action != "unchanged":
                body: dict[str, Any] = {
                    "external_id": external_id,
                    str(self._settings.es_content_field): content,
                    "content_sha256": content_sha,
                    "updated_at": _utc_now_iso(),
                }
                if existing_doc is None:
                    body["created_at"] = body["updated_at"]
                if item.source_id is not None:
                    body["source_id"] = str(item.source_id)
                if item.scope is not None:
                    body["scope"] = str(item.scope)
                if item.snapshot_id is not None:
                    body["snapshot_id"] = str(item.snapshot_id)
                if item.metadata is not None:
                    body["metadata"] = dict(item.metadata)
                if item.chunk_dedup_sha256 is not None:
                    body["chunk_dedup_sha256"] = str(item.chunk_dedup_sha256)
                if item.embedding is not None:
                    body[str(self._settings.es_embedding_field)] = list(item.embedding)
                ops.extend(
                    [
                        {
                            "index": {
                                "_index": str(self._settings.es_docs_index),
                                "_id": external_id,
                            }
                        },
                        body,
                    ]
                )
                if content_changed:
                    changed.append((DocId(external_id), content))
                    if existing_doc is not None:
                        updated_ids.append(DocId(external_id))

            results.append(
                ElasticDocsRepository.UpsertResult(
                    external_id=external_id,
                    id=DocId(external_id),
                    action=action,
                    content_changed=content_changed,
                )
            )

        if ops:
            self._client.bulk(ops)

        return results, changed, updated_ids

    def list_ids_by_external_id_prefix(self, prefix: str) -> list[tuple[DocId, str]]:
        if not str(prefix).strip():
            return []
        body = {
            "size": 1000,
            "_source": ["external_id"],
            "query": {"prefix": {"external_id": str(prefix)}},
            "sort": [{"external_id": "asc"}],
        }
        data = self._client.search(index=str(self._settings.es_docs_index), body=body)
        hits = (((data.get("hits") or {}).get("hits")) or [])
        out: list[tuple[DocId, str]] = []
        for hit in hits:
            source = dict(hit.get("_source") or {})
            ext_id = str(source.get("external_id") or hit.get("_id") or "")
            if ext_id:
                out.append((DocId(ext_id), ext_id))
        return out

    def snapshot_by_ids(self, ids: Sequence[DocId]) -> list[dict[str, Any]]:
        docs = self.get(ids)
        return [
            {
                "id": str(doc.id),
                "external_id": doc.external_id,
                "content": doc.content,
                "source_id": doc.source_id,
                "scope": (doc.metadata or {}).get("scope"),
                "snapshot_id": (doc.metadata or {}).get("snapshot_id"),
                "metadata": {
                    key: value
                    for key, value in dict(doc.metadata or {}).items()
                    if key not in {"scope", "snapshot_id"}
                },
                "content_sha256": hashlib.sha256(doc.content.encode("utf-8")).hexdigest(),
            }
            for doc in docs
        ]

    def snapshot_by_external_ids(self, external_ids: Sequence[str]) -> list[dict[str, Any]]:
        docs = self._client.mget(
            index=str(self._settings.es_docs_index),
            ids=[str(x) for x in external_ids if str(x).strip()],
        )
        out: list[dict[str, Any]] = []
        for doc in docs:
            if not bool(doc.get("found")):
                continue
            source = dict(doc.get("_source") or {})
            out.append(
                {
                    "id": str(source.get("external_id") or doc.get("_id") or ""),
                    "external_id": str(source.get("external_id") or doc.get("_id") or ""),
                    "content": str(source.get(self._settings.es_content_field) or ""),
                    "source_id": source.get("source_id"),
                    "scope": source.get("scope"),
                    "snapshot_id": source.get("snapshot_id"),
                    "metadata": dict(source.get("metadata") or {}),
                    "content_sha256": source.get("content_sha256"),
                    "chunk_dedup_sha256": source.get("chunk_dedup_sha256"),
                }
            )
        return out

    def restore_from_snapshots(self, snapshots: Sequence[dict[str, Any]]) -> None:
        ops: list[dict[str, Any]] = []
        for snap in snapshots:
            external_id = str(snap.get("external_id") or snap.get("id") or "").strip()
            if not external_id:
                continue
            body = {
                "external_id": external_id,
                str(self._settings.es_content_field): str(snap.get("content") or ""),
                "source_id": snap.get("source_id"),
                "scope": snap.get("scope"),
                "snapshot_id": snap.get("snapshot_id"),
                "metadata": dict(snap.get("metadata") or {}),
                "content_sha256": snap.get("content_sha256"),
                "chunk_dedup_sha256": snap.get("chunk_dedup_sha256"),
                "updated_at": _utc_now_iso(),
            }
            ops.extend(
                [
                    {"index": {"_index": str(self._settings.es_docs_index), "_id": external_id}},
                    body,
                ]
            )
        if ops:
            self._client.bulk(ops)

    def hard_delete_by_external_ids(self, external_ids: Sequence[str]) -> None:
        ops: list[dict[str, Any]] = []
        for external_id in external_ids:
            ext = str(external_id).strip()
            if not ext:
                continue
            ops.append({"delete": {"_index": str(self._settings.es_docs_index), "_id": ext}})
            ops.append({"delete": {"_index": str(self._settings.es_tombstones_index), "_id": ext}})
        if ops:
            self._client.bulk(ops)

    def list_external_ids_by_scope(self, scope: str) -> list[str]:
        scope_s = str(scope).strip()
        if not scope_s:
            return []
        body = {
            "size": 1000,
            "_source": ["external_id"],
            "query": {"term": {"scope": scope_s}},
            "sort": [{"external_id": "asc"}],
        }
        data = self._client.search(index=str(self._settings.es_docs_index), body=body)
        hits = (((data.get("hits") or {}).get("hits")) or [])
        return [
            str((hit.get("_source") or {}).get("external_id") or hit.get("_id") or "")
            for hit in hits
            if str((hit.get("_source") or {}).get("external_id") or hit.get("_id") or "").strip()
        ]

    def delete_tombstones(self, external_ids: Sequence[str]) -> int:
        ext_ids = [str(x).strip() for x in external_ids if str(x).strip()]
        if not ext_ids:
            return 0
        existing = self.get_tombstoned_external_ids(ext_ids)
        if not existing:
            return 0
        ops = [
            {"delete": {"_index": str(self._settings.es_tombstones_index), "_id": ext}}
            for ext in sorted(existing)
        ]
        self._client.bulk(ops)
        return len(existing)

    def get_tombstoned_external_ids(self, external_ids: Sequence[str]) -> set[str]:
        docs = self._client.mget(
            index=str(self._settings.es_tombstones_index),
            ids=[str(x) for x in external_ids if str(x).strip()],
        )
        return {
            str(doc.get("_id"))
            for doc in docs
            if bool(doc.get("found")) and str(doc.get("_id") or "").strip()
        }

    def delete_by_external_ids(
        self, external_ids: Sequence[str]
    ) -> tuple[int, list[DocId], list[str], int]:
        ext_ids = [str(x).strip() for x in external_ids if str(x).strip()]
        if not ext_ids:
            return 0, [], [], 0

        docs = self._client.mget(index=str(self._settings.es_docs_index), ids=ext_ids)
        found = {
            str(doc.get("_id")): DocId(str(doc.get("_id")))
            for doc in docs
            if bool(doc.get("found")) and str(doc.get("_id") or "").strip()
        }
        missing = [ext for ext in ext_ids if ext not in found]
        already_ts = self.get_tombstoned_external_ids(ext_ids)
        to_tombstone = [ext for ext in ext_ids if ext not in already_ts]

        ops: list[dict[str, Any]] = [
            {"delete": {"_index": str(self._settings.es_docs_index), "_id": ext}}
            for ext in found
        ]
        now = _utc_now_iso()
        for ext in to_tombstone:
            ops.extend(
                [
                    {
                        "index": {
                            "_index": str(self._settings.es_tombstones_index),
                            "_id": ext,
                        }
                    },
                    {"external_id": ext, "deleted_at": now},
                ]
            )
        if ops:
            self._client.bulk(ops)
        return len(found), list(found.values()), missing, len(to_tombstone)

    def _scan_documents(self) -> list[DomainDocument]:
        docs: list[DomainDocument] = []
        search_after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                "size": 500,
                "query": {"match_all": {}},
                "sort": [{"external_id": "asc"}],
            }
            if search_after is not None:
                body["search_after"] = list(search_after)
            data = self._client.search(index=str(self._settings.es_docs_index), body=body)
            hits = (((data.get("hits") or {}).get("hits")) or [])
            if not hits:
                break
            docs.extend(self._to_domain_document(hit) for hit in hits)
            search_after = hits[-1].get("sort")
            if not search_after:
                break
        return docs

    def _to_domain_document(self, hit: Mapping[str, Any]) -> DomainDocument:
        source = dict(hit.get("_source") or {})
        external_id = str(source.get("external_id") or hit.get("_id") or "")
        metadata = dict(source.get("metadata") or {})
        if source.get("scope") is not None:
            metadata["scope"] = source.get("scope")
        if source.get("snapshot_id") is not None:
            metadata["snapshot_id"] = source.get("snapshot_id")
        return DomainDocument(
            id=DocId(external_id),
            content=str(source.get(self._settings.es_content_field) or ""),
            external_id=external_id,
            source_id=(str(source.get("source_id")) if source.get("source_id") is not None else None),
            metadata=metadata,
        )


class ElasticVectorRepo(VectorRepoPort):
    def __init__(
        self,
        *,
        settings_obj: Settings | None = None,
        client: ElasticClient | None = None,
        dim: int | None = None,
        **_: Any,
    ) -> None:
        self._settings = settings_obj or global_settings
        self._client = client or ElasticClient(settings_obj=self._settings)
        self._client.ensure_indices(embed_dim=dim)

    @property
    def ntotal(self) -> int:
        return self._client.count(
            index=str(self._settings.es_docs_index),
            query={"exists": {"field": str(self._settings.es_embedding_field)}},
        )

    def upsert(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
        self.apply_delta_atomic(delete_ids=(), upserts=list(zip(ids, vectors, strict=False)))

    def apply_delta_atomic(
        self,
        *,
        delete_ids: Sequence[DocId],
        upserts: Sequence[tuple[DocId, Sequence[float]]],
    ) -> None:
        ops: list[dict[str, Any]] = []
        for doc_id in delete_ids:
            ext = str(doc_id).strip()
            if not ext:
                continue
            ops.extend(
                [
                    {"update": {"_index": str(self._settings.es_docs_index), "_id": ext}},
                    {"script": {"source": f"ctx._source.remove('{self._settings.es_embedding_field}')"}},
                ]
            )
        for doc_id, vector in upserts:
            ext = str(doc_id).strip()
            if not ext:
                continue
            ops.extend(
                [
                    {"update": {"_index": str(self._settings.es_docs_index), "_id": ext}},
                    {"doc": {str(self._settings.es_embedding_field): list(vector)}},
                ]
            )
        if ops:
            self._client.bulk(ops)

    def delete(self, ids: Sequence[DocId]) -> int:
        count = len([x for x in ids if str(x).strip()])
        self.apply_delta_atomic(delete_ids=ids, upserts=())
        return count

    def rebuild(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
        self.apply_delta_atomic(delete_ids=(), upserts=list(zip(ids, vectors, strict=False)))

    def similar(self, vector: Sequence[float], k: int) -> Sequence[tuple[DocId, float]]:
        if k <= 0:
            return []
        body = {
            "size": int(k),
            "knn": {
                "field": str(self._settings.es_embedding_field),
                "query_vector": list(vector),
                "k": int(k),
                "num_candidates": max(int(k), int(self._settings.es_hybrid_vector_k)),
            },
            "_source": False,
        }
        data = self._client.search(index=str(self._settings.es_docs_index), body=body)
        hits = (((data.get("hits") or {}).get("hits")) or [])
        if not hits:
            return []
        scores = [float(hit.get("_score") or 0.0) for hit in hits]
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            normalized = [1.0] * len(scores)
        else:
            normalized = [(score - min_s) / (max_s - min_s) for score in scores]
        return [
            (DocId(str(hit.get("_id"))), score)
            for hit, score in zip(hits, normalized, strict=False)
            if str(hit.get("_id") or "").strip()
        ]

    def count_indexed_docs(self) -> int:
        return self.ntotal

    def lexical_search(self, query: str, *, k: int) -> list[tuple[DocId, float]]:
        if not query.strip() or k <= 0:
            return []
        body = {
            "size": int(k),
            "query": {"match": {str(self._settings.es_content_field): {"query": query}}},
            "_source": False,
        }
        data = self._client.search(index=str(self._settings.es_docs_index), body=body)
        hits = (((data.get("hits") or {}).get("hits")) or [])
        if not hits:
            return []
        scores = [float(hit.get("_score") or 0.0) for hit in hits]
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            normalized = [1.0] * len(scores)
        else:
            normalized = [(score - min_s) / (max_s - min_s) for score in scores]
        return [
            (DocId(str(hit.get("_id"))), score)
            for hit, score in zip(hits, normalized, strict=False)
            if str(hit.get("_id") or "").strip()
        ]

    def get_mapping_dimension(self) -> int | None:
        mapping = self._client.get_mapping(index=str(self._settings.es_docs_index))
        props = (((mapping.get(str(self._settings.es_docs_index)) or {}).get("mappings") or {}).get("properties") or {})
        field = props.get(str(self._settings.es_embedding_field)) or {}
        dims = field.get("dims")
        return int(dims) if dims is not None else None


__all__ = ["ElasticBackendError", "ElasticDocsRepository", "ElasticVectorRepo"]
