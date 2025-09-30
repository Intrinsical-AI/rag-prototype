"""
Tests for RAG service input validation.

This module tests the critical bug fixes for input validation
in the RAG service to prevent crashes and ensure robust behavior.
"""

import pytest
from unittest.mock import Mock

from local_rag_backend.core.services.rag import RagService
from local_rag_backend.core.domain.entities import Document


class TestRagValidation:
    """Test input validation in RAG service."""

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever."""
        retriever = Mock()
        retriever.retrieve.return_value = (
            [Document(id=1, content="Test document")],
            [0.9]
        )
        return retriever

    @pytest.fixture
    def mock_generator(self):
        """Mock generator."""
        generator = Mock()
        generator.generate.return_value = "Generated answer"
        return generator

    @pytest.fixture
    def mock_history_storage(self):
        """Mock history storage."""
        return Mock()

    @pytest.fixture
    def rag_service(self, mock_retriever, mock_generator, mock_history_storage):
        """RAG service with mocked dependencies."""
        return RagService(mock_retriever, mock_generator, mock_history_storage)

    @pytest.mark.parametrize("invalid_question", [
        None,  # None question
        "",    # Empty string
        "   ", # Whitespace only
        "\t\n", # Tab and newline only
        "     \t   \n   ",  # Mixed whitespace
    ])
    def test_invalid_question_validation(self, rag_service, invalid_question):
        """Test that invalid questions raise ValueError."""
        with pytest.raises(ValueError, match="Question cannot be"):
            rag_service.ask(invalid_question, top_k=3)

    @pytest.mark.parametrize("invalid_question,expected_error", [
        (123, "Question must be a string, got int"),
        ([], "Question must be a string, got list"),
        ({}, "Question must be a string, got dict"),
        (object(), "Question must be a string, got object"),
    ])
    def test_non_string_question_validation(self, rag_service, invalid_question, expected_error):
        """Test that non-string questions raise appropriate ValueError."""
        with pytest.raises(ValueError, match=expected_error):
            rag_service.ask(invalid_question, top_k=3)  # type: ignore

    @pytest.mark.parametrize("invalid_top_k", [
        0,     # Zero
        -1,    # Negative
        -10,   # More negative
        51,    # Exceeds max limit
        100,   # Way over limit
    ])
    def test_invalid_top_k_validation(self, rag_service, invalid_top_k):
        """Test that invalid top_k values raise ValueError."""
        with pytest.raises(ValueError):
            rag_service.ask("Valid question", top_k=invalid_top_k)

    @pytest.mark.parametrize("invalid_top_k,expected_error", [
        (3.14, "top_k must be an integer, got float"),
        ("5", "top_k must be an integer, got str"),
        ([], "top_k must be an integer, got list"),
        (None, "top_k must be an integer, got NoneType"),
    ])
    def test_non_integer_top_k_validation(self, rag_service, invalid_top_k, expected_error):
        """Test that non-integer top_k values raise appropriate ValueError."""
        with pytest.raises(ValueError, match=expected_error):
            rag_service.ask("Valid question", top_k=invalid_top_k)  # type: ignore

    @pytest.mark.parametrize("valid_top_k", [1, 3, 5, 10, 25, 50])
    def test_valid_top_k_range(self, rag_service, valid_top_k):
        """Test that valid top_k values are accepted."""
        result = rag_service.ask("Valid question", top_k=valid_top_k)
        
        # Should not raise and should return proper structure
        assert isinstance(result, dict)
        assert "answer" in result
        assert "docs" in result
        assert "scores" in result

    @pytest.mark.parametrize("question,expected_sanitized", [
        ("  hello world  ", "hello world"),
        ("\ttest question\n", "test question"),
        ("   mixed   spaces   ", "mixed   spaces"),
        ("normal question", "normal question"),
    ])
    def test_question_sanitization(self, rag_service, mock_retriever, question, expected_sanitized):
        """Test that questions are properly sanitized."""
        rag_service.ask(question, top_k=3)
        
        # Verify retriever was called with sanitized question
        mock_retriever.retrieve.assert_called_once_with(expected_sanitized, 3)

    def test_successful_ask_flow_with_validation(self, rag_service, mock_retriever, mock_generator, mock_history_storage):
        """Test successful ask flow after validation."""
        question = "  What is the answer?  "  # With whitespace
        top_k = 5
        
        result = rag_service.ask(question, top_k=top_k)
        
        # Verify validation and sanitization occurred
        mock_retriever.retrieve.assert_called_once_with("What is the answer?", 5)
        mock_generator.generate.assert_called_once_with("What is the answer?", ["Test document"])
        mock_history_storage.save.assert_called_once_with("What is the answer?", "Generated answer", [1])
        
        # Verify result structure
        assert result == {
            "answer": "Generated answer",
            "docs": [Document(id=1, content="Test document")],
            "scores": [0.9]
        }

    def test_ask_with_no_documents_found(self, rag_service, mock_retriever, mock_history_storage):
        """Test ask behavior when no documents are found."""
        # Mock retriever to return empty results
        mock_retriever.retrieve.return_value = ([], [])
        
        result = rag_service.ask("Valid question", top_k=3)
        
        # Should return default message
        assert result["answer"] == "No hay documentos indexados para responder a tu pregunta."
        assert result["docs"] == []
        assert result["scores"] == []
        
        # Should still save to history
        mock_history_storage.save.assert_called_once_with("Valid question", result["answer"], [])

    def test_validation_methods_directly(self, rag_service):
        """Test validation methods directly."""
        # Test question validation
        assert rag_service._validate_and_sanitize_question("  valid  ") == "valid"
        
        with pytest.raises(ValueError, match="Question cannot be None"):
            rag_service._validate_and_sanitize_question(None)
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            rag_service._validate_and_sanitize_question("   ")
        
        # Test top_k validation
        assert rag_service._validate_top_k(5) == 5
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            rag_service._validate_top_k(0)
        
        with pytest.raises(ValueError, match="top_k cannot exceed 50"):
            rag_service._validate_top_k(51)

    def test_validation_error_propagation(self, rag_service):
        """Test that validation errors are properly propagated."""
        # Question validation error
        with pytest.raises(ValueError) as exc_info:
            rag_service.ask(None, top_k=3)
        assert "Question cannot be None" in str(exc_info.value)
        
        # top_k validation error
        with pytest.raises(ValueError) as exc_info:
            rag_service.ask("Valid question", top_k=0)
        assert "top_k must be positive" in str(exc_info.value)

    def test_validation_preserves_original_behavior(self, rag_service, mock_retriever, mock_generator, mock_history_storage):
        """Test that validation doesn't break original functionality."""
        # Test with typical valid inputs
        question = "What is machine learning?"
        top_k = 3
        
        result = rag_service.ask(question, top_k)
        
        # All components should be called as before
        mock_retriever.retrieve.assert_called_once_with(question, top_k)
        mock_generator.generate.assert_called_once()
        mock_history_storage.save.assert_called_once()
        
        # Result structure should be unchanged
        assert "answer" in result
        assert "docs" in result
        assert "scores" in result

    def test_edge_case_unicode_questions(self, rag_service, mock_retriever):
        """Test handling of unicode and special character questions."""
        unicode_questions = [
            "¿Qué es la inteligencia artificial?",
            "What is AI? 🤖",
            "机器学习是什么？",
            "Что такое ИИ?",
        ]
        
        for question in unicode_questions:
            result = rag_service.ask(question, top_k=3)
            
            # Should handle unicode gracefully
            assert isinstance(result, dict)
            mock_retriever.retrieve.assert_called_with(question, 3)

    def test_max_top_k_boundary(self, rag_service):
        """Test the exact boundary for max top_k."""
        # Should accept exactly 50
        result = rag_service.ask("Valid question", top_k=50)
        assert isinstance(result, dict)
        
        # Should reject 51
        with pytest.raises(ValueError, match="top_k cannot exceed 50"):
            rag_service.ask("Valid question", top_k=51)

    def test_validation_performance_impact(self, rag_service, mock_retriever):
        """Test that validation doesn't significantly impact performance."""
        # This is a basic test - in real scenarios you'd measure timing
        question = "Performance test question"
        
        # Multiple calls should all succeed
        for _ in range(10):
            result = rag_service.ask(question, top_k=5)
            assert isinstance(result, dict)
        
        # Verify all calls went through
        assert mock_retriever.retrieve.call_count == 10
