"""Elasticsearch/OpenSearch retrieval adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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
from local_rag_backend.core.ports import EmbedderPort


def _resolve_auth(
    *,
    username: str | None,
    password: str | None,
) -> tuple[str, str] | None:
    if username and password:
        return (str(username), str(password))
    return None


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(float(score) - min_score) / (max_score - min_score) for score in scores]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class ElasticLikeSearchRetriever:
    """Structured retrieval over Elasticsearch/OpenSearch style APIs."""

    def __init__(
        self,
        *,
        backend_name: str,
        base_url: str,
        docs_index: str,
        content_field: str,
        embedding_field: str,
        request_timeout_s: float,
        verify_tls: bool,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        embedder: EmbedderPort | None = None,
        dense_candidate_k: int = 50,
        client: httpx.Client | None = None,
    ) -> None:
        self._backend_name = str(backend_name)
        self._docs_index = str(docs_index)
        self._content_field = str(content_field)
        self._embedding_field = str(embedding_field)
        self._embedder = embedder
        self._dense_candidate_k = int(dense_candidate_k)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"ApiKey {api_key}"
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=float(request_timeout_s),
            verify=bool(verify_tls),
            auth=_resolve_auth(username=username, password=password),
            headers=headers,
        )

    def _filter_clauses(self, filters: Sequence[RetrievalFilter]) -> list[dict[str, Any]]:
        clauses: list[dict[str, Any]] = []
        for filter_item in filters:
            field_name: str
            if filter_item.field in {"scope", "snapshot_id", "source_id"}:
                field_name = filter_item.field
            else:
                metadata_key = metadata_key_for_filter_field(filter_item.field)
                if metadata_key is None:
                    raise ValueError(f"Unsupported filter field: {filter_item.field}")
                field_name = f"metadata.{metadata_key}.keyword"
            clauses.append({"terms": {field_name: list(filter_item.values)}})
        return clauses

    def _to_document(self, hit: Mapping[str, Any]) -> Document:
        source = dict(hit.get("_source") or {})
        metadata = dict(source.get("metadata") or {})
        if source.get("scope") is not None:
            metadata["scope"] = source.get("scope")
        if source.get("snapshot_id") is not None:
            metadata["snapshot_id"] = source.get("snapshot_id")
        external_id = str(source.get("external_id") or hit.get("_id") or "")
        return Document(
            id=DocId(external_id),
            content=str(source.get(self._content_field) or ""),
            external_id=external_id or None,
            source_id=(
                str(source.get("source_id")) if source.get("source_id") is not None else None
            ),
            metadata=metadata or None,
        )

    def _search(self, body: Mapping[str, Any]) -> list[RetrievedDoc]:
        response = self._client.post(f"/{self._docs_index}/_search", json=dict(body))
        response.raise_for_status()
        hits = ((response.json().get("hits") or {}).get("hits")) or []
        raw_scores = [float(hit.get("_score") or 0.0) for hit in hits]
        normalized_scores = _normalize_scores(raw_scores)
        return [
            RetrievedDoc(
                document=self._to_document(hit),
                score=float(score),
            )
            for hit, score in zip(hits, normalized_scores, strict=False)
        ]

    def _search_sparse(self, request: RetrievalRequest) -> RetrievalResult:
        filter_clauses = self._filter_clauses(request.filters)
        body: dict[str, Any] = {
            "size": int(request.top_k),
            "query": {
                "bool": {
                    "must": [{"match": {self._content_field: {"query": request.query}}}],
                    "filter": filter_clauses,
                }
            },
        }
        items = [
            RetrievedDoc(document=item.document, score=item.score, stage="sparse")
            for item in self._search(body)
        ]
        return RetrievalResult(
            items=tuple(items),
            mode_used="sparse",
            backend_used=self._backend_name,
            candidate_count=len(items),
        )

    def _search_dense(self, request: RetrievalRequest) -> RetrievalResult:
        if self._embedder is None:
            raise RuntimeError(f"{self._backend_name} dense retrieval requires an embedder")
        query_embedding = self._embedder.embed([request.query])[0]
        k = max(int(request.top_k), int(request.candidate_k or self._dense_candidate_k))
        knn: dict[str, Any] = {
            "field": self._embedding_field,
            "query_vector": list(query_embedding),
            "k": k,
            "num_candidates": k,
        }
        filter_clauses = self._filter_clauses(request.filters)
        if filter_clauses:
            knn["filter"] = {"bool": {"filter": filter_clauses}}
        body = {"size": int(request.top_k), "knn": knn}
        items = [
            RetrievedDoc(document=item.document, score=item.score, stage="dense")
            for item in self._search(body)
        ]
        if request.min_score is not None:
            items = [item for item in items if float(item.score) >= float(request.min_score)]
        return RetrievalResult(
            items=tuple(items[: request.top_k]),
            mode_used="dense",
            backend_used=self._backend_name,
            candidate_count=len(items),
        )

    def _search_dual(self, request: RetrievalRequest) -> RetrievalResult:
        if self._embedder is None:
            raise RuntimeError(f"{self._backend_name} dual retrieval requires an embedder")
        sparse_request = RetrievalRequest(
            query=request.query,
            top_k=max(int(request.top_k), int(request.dual_candidate_k or request.top_k)),
            mode="sparse",
            filters=request.filters,
        )
        sparse_result = self._search_sparse(sparse_request)
        if not sparse_result.items:
            return RetrievalResult(items=(), mode_used="dual", backend_used=self._backend_name)
        query_embedding = self._embedder.embed([request.query])[0]
        candidate_docs = [item.document for item in sparse_result.items]
        candidate_embeddings = self._embedder.embed([doc.content for doc in candidate_docs])
        reranked = sorted(
            (
                RetrievedDoc(
                    document=doc,
                    score=_cosine_similarity(query_embedding, candidate_embedding),
                    stage="dual_dense_rerank",
                    score_breakdown={"sparse_score": float(sparse_item.score)},
                )
                for doc, candidate_embedding, sparse_item in zip(
                    candidate_docs, candidate_embeddings, sparse_result.items, strict=False
                )
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        if request.min_score is not None:
            reranked = [item for item in reranked if float(item.score) >= float(request.min_score)]
        return RetrievalResult(
            items=tuple(reranked[: request.top_k]),
            mode_used="dual",
            backend_used=self._backend_name,
            candidate_count=len(candidate_docs),
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if request.mode == "sparse":
            return self._search_sparse(request)
        if request.mode == "dense":
            return self._search_dense(request)
        if request.mode == "dual":
            return self._search_dual(request)
        raise ValueError(f"Unsupported retrieval mode for {self._backend_name}: {request.mode}")
