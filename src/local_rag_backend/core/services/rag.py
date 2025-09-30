"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Module: RAG Service
Purpose: Core service orchestrating the retrieval-augmented generation pipeline.
         Coordinates document retrieval, answer generation, and history management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_rag_backend.core.ports import GeneratorPort, QAHistoryPort, RetrieverPort


class RagService:
    """Core service orchestrating the retrieval-augmented generation process.

    This service implements the main RAG workflow by coordinating three key components:
    - Document retrieval based on semantic similarity
    - Answer generation using retrieved context
    - Interaction history management for audit and analysis

    The service follows the hexagonal architecture pattern, depending only on port
    abstractions rather than concrete implementations.
    """

    def __init__(
        self, retriever: RetrieverPort, generator: GeneratorPort, history_storage: QAHistoryPort
    ):
        """Initialize the RAG service with required dependencies.

        Args:
            retriever: Port for document retrieval operations
            generator: Port for answer generation using LLMs
            history_storage: Port for persisting Q&A interactions
        """
        self.retriever = retriever
        self.generator = generator
        self.history_storage = history_storage

    def ask(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Process a question through the complete RAG pipeline.

        This method implements the core RAG workflow:
        1. Retrieve relevant documents using semantic similarity
        2. Generate an answer based on retrieved context
        3. Store the interaction in history for audit purposes

        Args:
            question: The user's question to be answered
            top_k: Maximum number of documents to retrieve for context

        Returns:
            Dictionary containing:
                - answer: Generated response text
                - docs: List of source documents used
                - scores: Relevance scores for each document

        Note:
            If no documents are found, returns a default message in Spanish
            to maintain consistency with the application's language.
        """
        # --- Document Retrieval Phase ---
        docs, scores = self.retriever.retrieve(question, top_k)

        # --- Handle Empty Results ---
        if not docs:
            answer = "No hay documentos indexados para responder a tu pregunta."
            self.history_storage.save(question, answer, [])
            return {"answer": answer, "docs": [], "scores": []}

        # --- Answer Generation Phase ---
        contexts = [doc.content for doc in docs]
        answer = self.generator.generate(question, contexts)

        # --- History Management ---
        source_ids = [doc.id for doc in docs]
        self.history_storage.save(question, answer, source_ids)

        return {"answer": answer, "docs": docs, "scores": scores}
