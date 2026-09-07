# src/core/rag.py
"""
RAG service for retrieval-augmented generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from local_rag_backend.core.domain.retrieval import (
    RetrievalFilter,
    RetrievalRequest,
)

if TYPE_CHECKING:
    from local_rag_backend.core.ports import GeneratorPort, QAHistoryPort, RetrieverPort

logger = logging.getLogger(__name__)

NO_DOCS_ANSWER = "No documents are indexed to answer your question."


class RagService:
    """Orchestrates the retrieval-augmented generation process."""

    def __init__(
        self,
        retriever: RetrieverPort,
        generator: GeneratorPort,
        history_storage: QAHistoryPort,
        no_docs_answer: str = NO_DOCS_ANSWER,
    ):
        self.retriever = retriever
        self.generator = generator
        self.history_storage = history_storage
        self.no_docs_answer = no_docs_answer

    def ask(
        self,
        question: str,
        top_k: int = 3,
        *,
        filters: tuple[RetrievalFilter, ...] = (),
        candidate_k: int | None = None,
        dual_candidate_k: int | None = None,
        retrieval_mode: str = "sparse",
    ) -> dict[str, Any]:
        """Processes a question through the RAG pipeline.

        1. Retrieves relevant documents.
        2. Generates an answer based on the documents.
        3. Stores the interaction in history.

        Returns:
            A dictionary containing the answer, source documents, and scores.
        """
        request = RetrievalRequest(
            query=question,
            top_k=top_k,
            mode=str(retrieval_mode),  # type: ignore[arg-type]
            filters=filters,
            candidate_k=candidate_k,
            dual_candidate_k=dual_candidate_k,
        )
        retrieval = self.retriever.retrieve(request)
        docs = list(retrieval.documents)
        scores = list(retrieval.scores)
        if len(docs) != len(scores):
            raise RuntimeError(
                f"Retriever contract violated: {len(docs)} docs != {len(scores)} scores."
            )

        # 2. Handle empty retrieval results
        if not docs:
            answer = self.no_docs_answer
            try:
                self.history_storage.save(question, answer, [])
            except Exception as e:  # pragma: no cover
                logger.warning("History persistence failed (ignored): %s", e)
            return {"answer": answer, "docs": [], "scores": [], "retrieval": retrieval}

        # 3. Generate an answer using the retrieved contexts
        contexts = [doc.content for doc in docs]
        answer = self.generator.generate(question, contexts)

        # 4. Record the interaction in history
        source_ids = [doc.id for doc in docs]
        try:
            self.history_storage.save(question, answer, source_ids)
        except Exception as e:  # pragma: no cover
            logger.warning("History persistence failed (ignored): %s", e)

        return {
            "answer": answer,
            "docs": docs,
            "scores": scores,
            "retrieval": retrieval,
        }
