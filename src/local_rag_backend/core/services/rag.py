# src/core/rag.py
"""
RAG service for retrieval-augmented generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_rag_backend.core.ports import GeneratorPort, QAHistoryPort, RetrieverPort

logger = logging.getLogger(__name__)


class RagService:
    """Orchestrates the retrieval-augmented generation process."""

    def __init__(
        self, retriever: RetrieverPort, generator: GeneratorPort, history_storage: QAHistoryPort
    ):
        self.retriever = retriever
        self.generator = generator
        self.history_storage = history_storage

    def ask(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Processes a question through the RAG pipeline.

        1. Retrieves relevant documents.
        2. Generates an answer based on the documents.
        3. Stores the interaction in history.

        Returns:
            A dictionary containing the answer, source documents, and scores.
        """
        # 1. Retrieve relevant documents
        docs, scores = self.retriever.retrieve(question, top_k)

        # 2. Handle empty retrieval results
        if not docs:
            answer = "No hay documentos indexados para responder a tu pregunta."
            try:
                self.history_storage.save(question, answer, [])
            except Exception as e:  # pragma: no cover
                logger.warning("History persistence failed (ignored): %s", e)
            return {"answer": answer, "docs": [], "scores": []}

        # 3. Generate an answer using the retrieved contexts
        contexts = [doc.content for doc in docs]
        answer = self.generator.generate(question, contexts)

        # 4. Record the interaction in history
        source_ids = [doc.id for doc in docs]
        try:
            self.history_storage.save(question, answer, source_ids)
        except Exception as e:  # pragma: no cover
            logger.warning("History persistence failed (ignored): %s", e)

        return {"answer": answer, "docs": docs, "scores": scores}
