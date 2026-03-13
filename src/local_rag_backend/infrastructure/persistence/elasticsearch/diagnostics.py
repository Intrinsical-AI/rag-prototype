"""Health/readiness diagnostics for Elasticsearch backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_rag_backend.infrastructure.persistence.elasticsearch.client import ElasticClient
from local_rag_backend.infrastructure.persistence.elasticsearch.document_storage import (
    ElasticDocsRepository,
    ElasticVectorRepo,
)
from local_rag_backend.infrastructure.persistence.elasticsearch.history_storage import (
    ElasticHistoryStorage,
)
from local_rag_backend.settings import Settings, settings as global_settings


class ElasticHealthDiagnostics:
    def __init__(
        self,
        *,
        settings_obj: Settings | None = None,
        client: ElasticClient | None = None,
    ) -> None:
        self._settings = settings_obj or global_settings
        self._client = client or ElasticClient(settings_obj=self._settings)
        self._docs = ElasticDocsRepository(settings_obj=self._settings, client=self._client)
        self._vector = ElasticVectorRepo(settings_obj=self._settings, client=self._client)
        self._history = ElasticHistoryStorage(settings_obj=self._settings, client=self._client)

    def ping_database(self) -> None:
        self._client.request_json("GET", "/", expected=(200,))

    def get_documents_count(self) -> int:
        return len(self._docs.get_all_documents())

    def get_history_count(self) -> int:
        return self._client.count(index=str(self._settings.es_history_index))

    def get_document_ids(self) -> tuple[str, ...]:
        return tuple(str(doc.id) for doc in self._docs.get_all_documents())

    def get_index_ids(self, *, id_map_path: str) -> tuple[str, ...]:
        _ = id_map_path
        body = {
            "size": 1000,
            "_source": False,
            "query": {"exists": {"field": str(self._settings.es_embedding_field)}},
            "sort": [{"external_id": "asc"}],
        }
        data = self._client.search(index=str(self._settings.es_docs_index), body=body)
        hits = (((data.get("hits") or {}).get("hits")) or [])
        return tuple(str(hit.get("_id")) for hit in hits if str(hit.get("_id") or "").strip())

    def get_retrieval_index_stats(
        self,
        *,
        index_path: str,
        id_map_path: str,
        vector_backend: str,
        dim: int | None = None,
        expected_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (index_path, id_map_path, vector_backend, dim, expected_manifest)
        mapping_dim = self._vector.get_mapping_dimension()
        docs_count = self.get_documents_count()
        vector_count = self._vector.count_indexed_docs()
        return {
            "status": "ok",
            "backend": "elasticsearch",
            "index_path": str(self._settings.es_docs_index),
            "id_map_path": str(self._settings.es_docs_index),
            "manifest_path": None,
            "dim": mapping_dim,
            "vectors": vector_count,
            "id_map_len": vector_count,
            "unique_ids": vector_count,
            "duplicates": 0,
            "documents": docs_count,
        }

    def get_incomplete_mutation_records_count(self, *, coordination_dir: Path) -> int:
        _ = coordination_dir
        return 0


def purge_index_artifacts_noop(*, index_path: str, id_map_path: str) -> None:
    _ = (index_path, id_map_path)
    return None


__all__ = ["ElasticHealthDiagnostics", "purge_index_artifacts_noop"]
