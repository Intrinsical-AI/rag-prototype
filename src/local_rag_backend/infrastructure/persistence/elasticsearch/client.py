"""Thin HTTP client for Elasticsearch/OpenSearch-like backends."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import httpx

from local_rag_backend.settings import Settings, settings as global_settings


class ElasticBackendError(RuntimeError):
    """Backend operation failed."""


class ElasticClient:
    def __init__(
        self,
        *,
        settings_obj: Settings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings_obj or global_settings
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=str(self._settings.es_base_url or ""),
            timeout=float(self._settings.es_request_timeout_s),
            verify=bool(self._settings.es_verify_tls),
            auth=self._resolve_auth(),
            headers=self._resolve_headers(),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _resolve_auth(self) -> tuple[str, str] | None:
        if self._settings.es_username and self._settings.es_password:
            return (str(self._settings.es_username), str(self._settings.es_password))
        return None

    def _resolve_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.es_api_key:
            headers["Authorization"] = f"ApiKey {self._settings.es_api_key}"
        return headers

    @staticmethod
    def _is_index_already_exists_error(exc: ElasticBackendError) -> bool:
        message = str(exc)
        return (
            "resource_already_exists_exception" in message or "already_exists_exception" in message
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        content: bytes | str | None = None,
        params: Mapping[str, Any] | None = None,
        expected: Sequence[int] = (200,),
    ) -> httpx.Response:
        response = self._client.request(
            method,
            path,
            json=json_body,
            content=content,
            params=params,
        )
        if response.status_code not in expected:
            raise ElasticBackendError(
                f"{method} {path} failed with status {response.status_code}: {response.text}"
            )
        return response

    def request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        content: bytes | str | None = None,
        params: Mapping[str, Any] | None = None,
        expected: Sequence[int] = (200,),
    ) -> dict[str, Any]:
        response = self._request(
            method,
            path,
            json_body=json_body,
            content=content,
            params=params,
            expected=expected,
        )
        if not response.content:
            return {}
        return dict(response.json())

    def head_ok(self, path: str) -> bool:
        response = self._client.request("HEAD", path)
        if response.status_code == 404:
            return False
        if response.status_code >= 400:
            raise ElasticBackendError(
                f"HEAD {path} failed with status {response.status_code}: {response.text}"
            )
        return True

    def bulk(self, operations: Iterable[dict[str, Any]]) -> dict[str, Any]:
        lines = [json.dumps(op) for op in operations]
        payload = "\n".join(lines) + ("\n" if lines else "")
        result = self.request_json(
            "POST",
            "/_bulk",
            content=payload.encode("utf-8"),
            params={"refresh": "true"},
            expected=(200,),
        )
        failures = []
        for item in result.get("items", []):
            for operation, outcome in item.items():
                status = int(outcome.get("status", 0))
                # Deleting an already absent document/tombstone is an idempotent success.
                missing_delete = (
                    operation == "delete" and status == 404 and not outcome.get("error")
                )
                if outcome.get("error") or (status >= 400 and not missing_delete):
                    failures.append(f"{operation} {outcome.get('_id', '?')}: status {status}")
        if failures or result.get("errors"):
            raise ElasticBackendError(
                "Elasticsearch bulk item failures: " + ("; ".join(failures[:10]) or "unknown items")
            )
        return result

    def mget(self, *, index: str, ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        payload = {"ids": [str(x) for x in ids]}
        data = self.request_json("POST", f"/{index}/_mget", json_body=payload)
        docs = data.get("docs")
        if not isinstance(docs, list):
            return []
        return [dict(doc) for doc in docs if isinstance(doc, dict)]

    def search(
        self,
        *,
        index: str,
        body: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.request_json("POST", f"/{index}/_search", json_body=dict(body), params=params)

    def count(self, *, index: str, query: Mapping[str, Any] | None = None) -> int:
        body = {"query": dict(query or {"match_all": {}})}
        data = self.request_json("POST", f"/{index}/_count", json_body=body)
        return int(data.get("count") or 0)

    def delete(self, *, index: str, id_: str) -> None:
        self._request("DELETE", f"/{index}/_doc/{id_}", expected=(200, 202, 404))

    def index_document(self, *, index: str, id_: str, body: Mapping[str, Any]) -> None:
        self._request(
            "PUT",
            f"/{index}/_doc/{id_}",
            json_body=dict(body),
            params={"refresh": "true"},
            expected=(200, 201),
        )

    def get_mapping(self, *, index: str) -> dict[str, Any]:
        return self.request_json("GET", f"/{index}/_mapping")

    def ensure_indices(self, *, embed_dim: int | None) -> None:
        docs_index = str(self._settings.es_docs_index)
        history_index = str(self._settings.es_history_index)
        system_index = str(self._settings.es_system_index)
        tombstones_index = str(self._settings.es_tombstones_index)

        self._ensure_docs_index(index=docs_index, embed_dim=embed_dim)
        self._ensure_history_index(index=history_index)
        self._ensure_system_index(index=system_index)
        self._ensure_tombstones_index(index=tombstones_index)

    def _ensure_docs_index(self, *, index: str, embed_dim: int | None) -> None:
        if self.head_ok(f"/{index}"):
            mapping = self.get_mapping(index=index)
            props = ((mapping.get(index) or {}).get("mappings") or {}).get("properties") or {}
            properties_to_add: dict[str, Any] = {
                field_name: field_mapping
                for field_name, field_mapping in {
                    "scope": {"type": "keyword"},
                    "snapshot_id": {"type": "keyword"},
                }.items()
                if field_name not in props
            }

            if embed_dim is not None:
                embedding = props.get(str(self._settings.es_embedding_field)) or {}
                dims = embedding.get("dims")
                if dims is not None and int(dims) != int(embed_dim):
                    raise ElasticBackendError(
                        f"Elasticsearch index {index!r} embedding dims mismatch: {dims} != {embed_dim}"
                    )
                if dims is None:
                    properties_to_add[str(self._settings.es_embedding_field)] = {
                        "type": "dense_vector",
                        "dims": int(embed_dim),
                        "index": True,
                        "similarity": "cosine",
                    }

            if properties_to_add:
                self.request_json(
                    "PUT",
                    f"/{index}/_mapping",
                    json_body={"properties": properties_to_add},
                    expected=(200,),
                )
            return

        properties: dict[str, Any] = {
            "external_id": {"type": "keyword"},
            "source_id": {"type": "keyword"},
            "scope": {"type": "keyword"},
            "snapshot_id": {"type": "keyword"},
            str(self._settings.es_content_field): {"type": "text"},
            "metadata": {"type": "object", "dynamic": True},
            "content_sha256": {"type": "keyword"},
            "chunk_dedup_sha256": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        }
        if embed_dim is not None:
            properties[str(self._settings.es_embedding_field)] = {
                "type": "dense_vector",
                "dims": int(embed_dim),
                "index": True,
                "similarity": "cosine",
            }
        payload = {"mappings": {"dynamic": True, "properties": properties}}
        try:
            self.request_json("PUT", f"/{index}", json_body=payload, expected=(200,))
        except ElasticBackendError as exc:
            if self._is_index_already_exists_error(exc):
                return
            raise

    def _ensure_history_index(self, *, index: str) -> None:
        if self.head_ok(f"/{index}"):
            return
        payload = {
            "mappings": {
                "dynamic": False,
                "properties": {
                    "id": {"type": "long"},
                    "question": {"type": "text"},
                    "answer": {"type": "text"},
                    "source_ids": {"type": "keyword"},
                    "created_at": {"type": "date"},
                },
            }
        }
        try:
            self.request_json("PUT", f"/{index}", json_body=payload, expected=(200,))
        except ElasticBackendError as exc:
            if self._is_index_already_exists_error(exc):
                return
            raise

    def _ensure_system_index(self, *, index: str) -> None:
        if self.head_ok(f"/{index}"):
            return
        payload = {
            "mappings": {
                "dynamic": False,
                "properties": {
                    "key": {"type": "keyword"},
                    "version": {"type": "long"},
                    "updated_at": {"type": "date"},
                },
            }
        }
        try:
            self.request_json("PUT", f"/{index}", json_body=payload, expected=(200,))
        except ElasticBackendError as exc:
            if self._is_index_already_exists_error(exc):
                return
            raise

    def _ensure_tombstones_index(self, *, index: str) -> None:
        if self.head_ok(f"/{index}"):
            return
        payload = {
            "mappings": {
                "dynamic": False,
                "properties": {
                    "external_id": {"type": "keyword"},
                    "deleted_at": {"type": "date"},
                },
            }
        }
        try:
            self.request_json("PUT", f"/{index}", json_body=payload, expected=(200,))
        except ElasticBackendError as exc:
            if self._is_index_already_exists_error(exc):
                return
            raise
