"""Local retrieval backend with explicit sparse/dense/dual modes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_rag_backend.core.domain.entities import Document
    from local_rag_backend.core.domain.retrieval import RetrievalFilter

from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
    document_matches_filters,
    retrieval_result_from_pairs,
)
from local_rag_backend.core.ports import DocumentRepoPort, EmbedderPort, VectorRepoPort
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class LocalSplitSearchRetriever:
    """Retrieval adapter for local SQL/vector stores."""

    def __init__(
        self,
        *,
        doc_repo: DocumentRepoPort,
        embedder: EmbedderPort | None = None,
        vector_repo: VectorRepoPort | None = None,
        preloaded_docs: Sequence[Document] | None = None,
    ) -> None:
        self._doc_repo = doc_repo
        self._embedder = embedder
        self._vector_repo = vector_repo
        self._preloaded_docs = tuple(preloaded_docs) if preloaded_docs is not None else None

    def _all_docs(self) -> tuple[Document, ...]:
        if self._preloaded_docs is not None:
            return self._preloaded_docs
        return tuple(self._doc_repo.get_all_documents())

    def _filtered_docs(self, filters: Sequence[RetrievalFilter]) -> tuple[Document, ...]:
        docs = self._all_docs()
        if not filters:
            return docs
        return tuple(doc for doc in docs if document_matches_filters(doc, filters))

    def _retrieve_sparse(self, request: RetrievalRequest) -> RetrievalResult:
        docs = self._filtered_docs(request.filters)
        if not docs:
            return RetrievalResult(items=(), mode_used="sparse", backend_used="local_split")
        retriever = SparseBM25Retriever(
            documents=[doc.content for doc in docs],
            doc_ids=[doc.id for doc in docs],
            doc_repo=self._doc_repo,
            preloaded_docs=docs,
        )
        raw_result = retriever.retrieve(request)
        legacy_docs = list(raw_result.documents)
        legacy_scores = list(raw_result.scores)
        return retrieval_result_from_pairs(
            docs=legacy_docs,
            scores=legacy_scores,
            mode_used="sparse",
            backend_used="local_split",
            stage="sparse",
            candidate_count=len(docs),
        )

    def _retrieve_dense(self, request: RetrievalRequest) -> RetrievalResult:
        if self._embedder is None or self._vector_repo is None:
            raise RuntimeError("Dense retrieval requires embedder and vector_repo")
        query_embedding = self._embedder.embed([request.query])[0]
        candidate_k = max(request.top_k, int(request.candidate_k or request.top_k))
        id_score_pairs = self._vector_repo.similar(query_embedding, candidate_k)
        if not id_score_pairs:
            return RetrievalResult(items=(), mode_used="dense", backend_used="local_split")
        doc_ids = [doc_id for doc_id, _score in id_score_pairs]
        docs = self._doc_repo.get(doc_ids)
        docs_by_id = {doc.id: doc for doc in docs}
        items: list[RetrievedDoc] = []
        for doc_id, score in id_score_pairs:
            doc = docs_by_id.get(doc_id)
            if doc is None or not document_matches_filters(doc, request.filters):
                continue
            if request.min_score is not None and float(score) < float(request.min_score):
                continue
            items.append(RetrievedDoc(document=doc, score=float(score), stage="dense"))
            if len(items) >= request.top_k:
                break
        return RetrievalResult(
            items=tuple(items),
            mode_used="dense",
            backend_used="local_split",
            candidate_count=len(id_score_pairs),
        )

    def _retrieve_dual(self, request: RetrievalRequest) -> RetrievalResult:
        if self._embedder is None:
            raise RuntimeError("Dual retrieval requires an embedder")
        sparse_request = RetrievalRequest(
            query=request.query,
            top_k=max(request.top_k, int(request.dual_candidate_k or request.top_k)),
            mode="sparse",
            filters=request.filters,
        )
        sparse_result = self._retrieve_sparse(sparse_request)
        if not sparse_result.items:
            return RetrievalResult(items=(), mode_used="dual", backend_used="local_split")
        query_embedding = self._embedder.embed([request.query])[0]
        candidate_docs = [item.document for item in sparse_result.items]
        candidate_embeddings = self._embedder.embed([doc.content for doc in candidate_docs])
        reranked = sorted(
            (
                RetrievedDoc(
                    document=doc,
                    score=_cosine_similarity(query_embedding, doc_embedding),
                    stage="dual_dense_rerank",
                    score_breakdown={"sparse_rank": float(index)},
                )
                for index, (doc, doc_embedding) in enumerate(
                    zip(candidate_docs, candidate_embeddings, strict=False)
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
            backend_used="local_split",
            candidate_count=len(candidate_docs),
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if request.mode == "sparse":
            return self._retrieve_sparse(request)
        if request.mode == "dense":
            return self._retrieve_dense(request)
        if request.mode == "dual":
            return self._retrieve_dual(request)
        raise ValueError(
            f"Unsupported retrieval mode for LocalSplitSearchRetriever: {request.mode}"
        )
