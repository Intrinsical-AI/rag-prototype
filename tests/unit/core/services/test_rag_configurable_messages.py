"""
Tests for configurable messages in RAG service.

This module tests the bug fix for hardcoded messages,
ensuring they are configurable and internationalizable.
"""

from unittest.mock import Mock, patch

import pytest

from local_rag_backend.core.services.rag import RagService
from local_rag_backend.settings import Settings


class TestRagConfigurableMessages:
    """Test configurable messages in RAG service."""

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever that returns empty results."""
        retriever = Mock()
        retriever.retrieve.return_value = ([], [])  # No documents found
        return retriever

    @pytest.fixture
    def mock_generator(self):
        """Mock generator."""
        return Mock()

    @pytest.fixture
    def mock_history_storage(self):
        """Mock history storage."""
        return Mock()

    @pytest.fixture
    def rag_service(self, mock_retriever, mock_generator, mock_history_storage):
        """RAG service with mocked dependencies."""
        return RagService(mock_retriever, mock_generator, mock_history_storage)

    def test_default_no_documents_message(self, rag_service, mock_history_storage):
        """Test that default no documents message is in English."""
        result = rag_service.ask("test question", top_k=3)

        # Should use the default English message
        expected_message = "No documents are available to answer your question."
        assert result["answer"] == expected_message

        # Should save to history with the configured message
        mock_history_storage.save.assert_called_once_with(
            "test question", expected_message, []
        )

    def test_custom_no_documents_message_via_settings(self, mock_retriever, mock_generator, mock_history_storage):
        """Test that custom message can be configured via settings."""
        # Create custom settings with different message
        custom_settings = Settings(
            no_documents_message="Custom message: No docs found!"
        )

        with patch('local_rag_backend.core.services.rag.settings', custom_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            assert result["answer"] == "Custom message: No docs found!"
            mock_history_storage.save.assert_called_once_with(
                "test question", "Custom message: No docs found!", []
            )

    def test_spanish_message_configuration(self, mock_retriever, mock_generator, mock_history_storage):
        """Test configuration with Spanish message for backward compatibility."""
        # Configure Spanish message like the original
        spanish_settings = Settings(
            no_documents_message="No hay documentos indexados para responder a tu pregunta."
        )

        with patch('local_rag_backend.core.services.rag.settings', spanish_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            assert result["answer"] == "No hay documentos indexados para responder a tu pregunta."

    def test_empty_message_configuration(self, mock_retriever, mock_generator, mock_history_storage):
        """Test that empty message is handled correctly."""
        empty_settings = Settings(no_documents_message="")

        with patch('local_rag_backend.core.services.rag.settings', empty_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            assert result["answer"] == ""

    def test_unicode_message_configuration(self, mock_retriever, mock_generator, mock_history_storage):
        """Test that unicode messages work correctly."""
        unicode_messages = [
            "没有找到相关文档。",  # Chinese
            "Документы не найдены.",  # Russian
            "ドキュメントが見つかりません。",  # Japanese
            "🔍 No documents found! 📄",  # With emojis
        ]

        for message in unicode_messages:
            unicode_settings = Settings(no_documents_message=message)

            with patch('local_rag_backend.core.services.rag.settings', unicode_settings):
                rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
                result = rag_service.ask("test question", top_k=3)

                assert result["answer"] == message

    def test_long_message_configuration(self, mock_retriever, mock_generator, mock_history_storage):
        """Test that long messages are handled correctly."""
        long_message = (
            "We apologize, but no documents are currently available in our database "
            "to provide an accurate answer to your question. Please try rephrasing "
            "your query or contact support for assistance."
        )

        long_settings = Settings(no_documents_message=long_message)

        with patch('local_rag_backend.core.services.rag.settings', long_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            assert result["answer"] == long_message

    def test_message_with_special_characters(self, mock_retriever, mock_generator, mock_history_storage):
        """Test messages with special characters and formatting."""
        special_messages = [
            "No documents found.\nPlease try again.",  # With newlines
            "Error: No docs available (code: 404)",  # With symbols
            "¡No hay documentos disponibles!",  # With Spanish punctuation
            "\"No documents\" - System Message",  # With quotes
        ]

        for message in special_messages:
            special_settings = Settings(no_documents_message=message)

            with patch('local_rag_backend.core.services.rag.settings', special_settings):
                rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
                result = rag_service.ask("test question", top_k=3)

                assert result["answer"] == message

    def test_message_configuration_via_environment(self, mock_retriever, mock_generator, mock_history_storage):
        """Test that message can be configured via environment variables."""
        env_message = "Environment configured message"

        with patch.dict('os.environ', {'NO_DOCUMENTS_MESSAGE': env_message}):
            # Create new settings instance to pick up environment variable
            env_settings = Settings()

            with patch('local_rag_backend.core.services.rag.settings', env_settings):
                rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
                result = rag_service.ask("test question", top_k=3)

                assert result["answer"] == env_message

    def test_message_consistency_across_calls(self, rag_service, mock_history_storage):
        """Test that message is consistent across multiple calls."""
        # Make multiple calls
        for i in range(3):
            result = rag_service.ask(f"test question {i}", top_k=3)
            assert result["answer"] == "No documents are available to answer your question."

        # All calls should have used the same message
        assert mock_history_storage.save.call_count == 3
        for call in mock_history_storage.save.call_args_list:
            assert call[0][1] == "No documents are available to answer your question."

    def test_message_not_affected_by_question_content(self, rag_service):
        """Test that message is not affected by question content."""
        questions = [
            "Simple question",
            "Question with special chars: !@#$%",
            "Very long question that goes on and on with lots of details",
            "Question with unicode: 你好世界",
            "",  # This will be caught by validation, but let's test anyway
        ]

        for question in questions:
            if question:  # Skip empty question as it will raise ValueError
                result = rag_service.ask(question, top_k=3)
                assert result["answer"] == "No documents are available to answer your question."

    def test_settings_validation_for_message(self):
        """Test that settings validation works for the message field."""
        # Test that message field accepts various string types
        valid_messages = [
            "Simple message",
            "",  # Empty string
            "Message with\nnewlines",
            "Unicode: 🌍",
        ]

        for message in valid_messages:
            settings = Settings(no_documents_message=message)
            assert settings.no_documents_message == message

    def test_backward_compatibility_with_original_behavior(self, mock_retriever, mock_generator, mock_history_storage):
        """Test that the fix maintains backward compatibility."""
        # The original hardcoded message should still work if configured
        original_message = "No hay documentos indexados para responder a tu pregunta."

        original_settings = Settings(no_documents_message=original_message)

        with patch('local_rag_backend.core.services.rag.settings', original_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            # Should work exactly like before
            assert result["answer"] == original_message
            assert result["docs"] == []
            assert result["scores"] == []

    def test_message_configuration_doesnt_affect_successful_queries(self, mock_generator, mock_history_storage):
        """Test that message configuration doesn't affect successful queries."""
        from local_rag_backend.core.domain.entities import Document

        # Mock retriever that returns documents
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = (
            [Document(id=1, content="Test document")],
            [0.9]
        )

        mock_generator.generate.return_value = "Generated answer"

        custom_settings = Settings(no_documents_message="Custom no docs message")

        with patch('local_rag_backend.core.services.rag.settings', custom_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            # Should use generated answer, not the configured message
            assert result["answer"] == "Generated answer"
            assert len(result["docs"]) == 1
            assert len(result["scores"]) == 1
