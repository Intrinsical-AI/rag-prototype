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

from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from local_rag_backend.core.ports import GeneratorPort, QAHistoryPort, RetrieverPort


class RagService:
    """Core service orchestrating the retrieval-augmented generation process.

    This service implements the main RAG workflow by coordinating three key components:
    - Document retrieval based on semantic similarity
    - Answer generation using retrieved context
    - Interaction history management for audit and analysis
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
        """Process a question through the complete RAG pipeline with robust validation.

        This method implements the core RAG workflow with comprehensive input validation:
        1. Validate and sanitize input parameters
        2. Retrieve relevant documents using semantic similarity
        3. Generate an answer based on retrieved context
        4. Store the interaction in history for audit purposes

        Args:
            question: The user's question to be answered
            top_k: Maximum number of documents to retrieve for context (1-50)

        Returns:
            Dictionary containing:
                - answer: Generated response text
                - docs: List of source documents used
                - scores: Relevance scores for each document

        Raises:
            ValueError: If question is invalid (None, empty, or whitespace-only)
            ValueError: If top_k is invalid (not positive or exceeds limits)

        Note:
            If no documents are found after validation, returns a configurable
            default message to maintain consistency.
        """
        # --- Input Validation ---
        validated_question = self._validate_and_sanitize_question(question)
        validated_top_k = self._validate_top_k(top_k)

        # --- Document Retrieval Phase ---
        docs, scores = self.retriever.retrieve(validated_question, validated_top_k)

        # --- Handle Empty Results ---
        if not docs:
            answer = settings.no_documents_message
            self._save_to_history_if_enabled(validated_question, answer, [])
            return {"answer": answer, "docs": [], "scores": []}

        # --- Answer Generation Phase ---
        contexts = [doc.content for doc in docs]
        answer = self.generator.generate(validated_question, contexts)

        # --- History Management ---
        source_ids = [doc.id for doc in docs]
        self._save_to_history_if_enabled(validated_question, answer, source_ids)

        return {"answer": answer, "docs": docs, "scores": scores}

    def _save_to_history_if_enabled(
        self, question: str, answer: str, source_ids: list[int]
    ) -> None:
        """Save interaction to history if enabled and manage size limits.

        Args:
            question: The user's question
            answer: The generated answer
            source_ids: IDs of source documents used
        """
        if not settings.enable_history:
            return

        # Save the interaction
        self.history_storage.save(question, answer, source_ids)

        # Manage history size if limit is set
        if settings.max_history_entries > 0:
            self._cleanup_history_if_needed()

    def _cleanup_history_if_needed(self) -> None:
        """Clean up old history entries if limit is exceeded.

        This is a best-effort cleanup. If the history storage doesn't support
        cleanup operations, this will be a no-op.
        """
        # Check if history storage supports cleanup operations
        if hasattr(self.history_storage, "cleanup_old_entries"):
            self.history_storage.cleanup_old_entries(settings.max_history_entries)
        elif hasattr(self.history_storage, "get_entry_count") and hasattr(
            self.history_storage, "delete_oldest_entries"
        ):
            try:
                current_count = self.history_storage.get_entry_count()
                if current_count > settings.max_history_entries:
                    excess = current_count - settings.max_history_entries
                    self.history_storage.delete_oldest_entries(excess)
            except Exception:  # noqa: S110
                # If cleanup fails, continue without error to maintain service availability
                # This is intentional to prevent history cleanup failures from breaking the service
                pass  # nosec B110

    def _validate_and_sanitize_question(self, question: str | None) -> str:
        """Validate and sanitize the input question.

        Args:
            question: Raw question input to validate

        Returns:
            Sanitized question string

        Raises:
            ValueError: If question is invalid
        """
        if question is None:
            raise ValueError("Question cannot be None")

        if not isinstance(question, str):
            raise ValueError(f"Question must be a string, got {type(question).__name__}")

        sanitized = question.strip()
        if not sanitized:
            raise ValueError("Question cannot be empty or whitespace-only")

        return sanitized

    def _validate_top_k(self, top_k: int) -> int:
        """Validate the top_k parameter.

        Args:
            top_k: Number of documents to retrieve

        Returns:
            Validated top_k value

        Raises:
            ValueError: If top_k is invalid
        """
        if not isinstance(top_k, int):
            raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        # Reasonable upper limit to prevent performance issues
        max_top_k = 50
        if top_k > max_top_k:
            raise ValueError(f"top_k cannot exceed {max_top_k}, got {top_k}")

        return top_k
