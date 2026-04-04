"""Solr retrieval adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import (
    RetrievalFilter,
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
    metadata_key_for_filter_field,
)
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.infrastructure.retrieval.scoring import normalize_min_max_scores


class SolrSearchRetriever:
    """Sparse-only retrieval against Solr."""

    def __init__(
        self,
        *,
        base_url: str,
        core: str,
        content_field: str,
        request_timeout_s: float,
        client: httpx.Client | None = None,
    ) -> None:
        self._core = str(core)
        self._content_field = str(content_field)
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=base_url, timeout=float(request_timeout_s))

    def _fq(self, filter_item: RetrievalFilter) -> str:
        if filter_item.field in {"scope", "snapshot_id", "source_id"}:
            field_name = filter_item.field
        else:
            metadata_key = metadata_key_for_filter_field(filter_item.field)
            field_name = f"metadata.{metadata_key}"
        encoded = " OR ".join(json.dumps(str(value)) for value in filter_item.values)
        if len(filter_item.values) == 1:
            return f"{field_name}:{encoded}"
        return f"{field_name}:({encoded})"

    def _to_document(self, row: Mapping[str, Any]) -> Document:
        metadata = dict(row.get("metadata") or {})
        metadata_json = row.get("metadata_json")
        if metadata_json and not metadata:
            try:
                parsed = json.loads(str(metadata_json))
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                metadata = {str(key): value for key, value in parsed.items()}
        for field in ("scope", "snapshot_id", "path", "language", "unit_type"):
            if row.get(field) is not None:
                metadata.setdefault(field, row.get(field))
        external_id = str(row.get("external_id") or row.get("id") or "")
        return Document(
            id=DocId(external_id),
            content=str(row.get(self._content_field) or ""),
            external_id=external_id or None,
            source_id=(str(row.get("source_id")) if row.get("source_id") is not None else None),
            metadata=metadata or None,
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if request.mode != "sparse":
            raise ValueError("SEARCH_BACKEND=solr supports only retrieval_mode=sparse in v1")
        params: list[tuple[str, str | int | float | bool | None]] = [
            ("q", request.query),
            ("df", self._content_field),
            ("rows", int(request.top_k)),
            ("wt", "json"),
        ]
        params.extend(("fq", self._fq(filter_item)) for filter_item in request.filters)
        response = self._client.get(f"/solr/{self._core}/select", params=params)
        response.raise_for_status()
        docs = ((response.json().get("response") or {}).get("docs")) or []
        raw_scores = [float(doc.get("score") or 0.0) for doc in docs]
        normalized_scores = normalize_min_max_scores(
            raw_scores,
            flat_value=0.0,
            singleton_value=1.0,
        )
        items = [
            RetrievedDoc(document=self._to_document(doc), score=float(score), stage="sparse")
            for doc, score in zip(docs, normalized_scores, strict=False)
        ]
        return RetrievalResult(
            items=tuple(items),
            mode_used="sparse",
            backend_used="solr",
            candidate_count=len(items),
        )
