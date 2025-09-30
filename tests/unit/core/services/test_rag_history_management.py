"""
Tests for RAG service history management and memory leak prevention.

This module tests the bug fix for unlimited history growth
and configurable history management features.
"""

from unittest.mock import Mock, patch

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.services.rag import RagService
from local_rag_backend.settings import Settings


class TestRagHistoryManagement:
    """Test history management and memory leak prevention in RAG service."""

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever that returns documents."""
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
        """Mock history storage with cleanup capabilities."""
        storage = Mock()
        storage.save = Mock()
        storage.cleanup_old_entries = Mock()
        storage.get_entry_count = Mock(return_value=5)
        storage.delete_oldest_entries = Mock()
        return storage

    @pytest.fixture
    def rag_service(self, mock_retriever, mock_generator, mock_history_storage):
        """RAG service with mocked dependencies."""
        return RagService(mock_retriever, mock_generator, mock_history_storage)

    def test_history_enabled_by_default(self, rag_service, mock_history_storage):
        """Test that history is enabled by default."""
        result = rag_service.ask("test question", top_k=3)

        # Should save to history
        mock_history_storage.save.assert_called_once()
        assert result["answer"] == "Generated answer"

    def test_history_disabled_via_settings(self, mock_retriever, mock_generator, mock_history_storage):
        """Test that history can be disabled via settings."""
        disabled_settings = Settings(enable_history=False)

        with patch('local_rag_backend.core.services.rag.settings', disabled_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            # Should NOT save to history
            mock_history_storage.save.assert_not_called()
            assert result["answer"] == "Generated answer"

    def test_history_cleanup_with_limit(self, mock_retriever, mock_generator, mock_history_storage):
        """Test that history cleanup is triggered when limit is set."""
        limited_settings = Settings(max_history_entries=100)

        with patch('local_rag_backend.core.services.rag.settings', limited_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            # Should save to history and trigger cleanup
            mock_history_storage.save.assert_called_once()
            mock_history_storage.cleanup_old_entries.assert_called_once_with(100)

    def test_history_no_cleanup_with_unlimited(self, mock_retriever, mock_generator, mock_history_storage):
        """Test that no cleanup occurs when max_history_entries is 0 (unlimited)."""
        unlimited_settings = Settings(max_history_entries=0)

        with patch('local_rag_backend.core.services.rag.settings', unlimited_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            # Should save to history but NOT trigger cleanup
            mock_history_storage.save.assert_called_once()
            mock_history_storage.cleanup_old_entries.assert_not_called()

    def test_history_alternative_cleanup_method(self, mock_retriever, mock_generator):
        """Test alternative cleanup method when primary method is not available."""
        # Mock storage without cleanup_old_entries but with alternative methods
        alt_storage = Mock()
        alt_storage.save = Mock()
        alt_storage.get_entry_count = Mock(return_value=150)  # Over limit
        alt_storage.delete_oldest_entries = Mock()
        # Explicitly remove cleanup_old_entries method
        del alt_storage.cleanup_old_entries

        limited_settings = Settings(max_history_entries=100)

        with patch('local_rag_backend.core.services.rag.settings', limited_settings):
            rag_service = RagService(mock_retriever, mock_generator, alt_storage)
            result = rag_service.ask("test question", top_k=3)

            # Should use alternative cleanup method
            alt_storage.save.assert_called_once()
            alt_storage.get_entry_count.assert_called_once()
            alt_storage.delete_oldest_entries.assert_called_once_with(50)  # 150 - 100

    def test_history_no_cleanup_when_under_limit(self, mock_retriever, mock_generator):
        """Test that no cleanup occurs when under the limit."""
        alt_storage = Mock()
        alt_storage.save = Mock()
        alt_storage.get_entry_count = Mock(return_value=50)  # Under limit
        alt_storage.delete_oldest_entries = Mock()
        # Explicitly remove cleanup_old_entries method
        del alt_storage.cleanup_old_entries

        limited_settings = Settings(max_history_entries=100)

        with patch('local_rag_backend.core.services.rag.settings', limited_settings):
            rag_service = RagService(mock_retriever, mock_generator, alt_storage)
            result = rag_service.ask("test question", top_k=3)

            # Should not trigger cleanup
            alt_storage.save.assert_called_once()
            alt_storage.get_entry_count.assert_called_once()
            alt_storage.delete_oldest_entries.assert_not_called()

    def test_history_cleanup_failure_graceful_handling(self, mock_retriever, mock_generator):
        """Test that cleanup failures don't break the service."""
        failing_storage = Mock()
        failing_storage.save = Mock()
        failing_storage.get_entry_count = Mock(side_effect=Exception("Cleanup failed"))

        limited_settings = Settings(max_history_entries=100)

        with patch('local_rag_backend.core.services.rag.settings', limited_settings):
            rag_service = RagService(mock_retriever, mock_generator, failing_storage)

            # Should not raise exception even if cleanup fails
            result = rag_service.ask("test question", top_k=3)
            assert result["answer"] == "Generated answer"
            failing_storage.save.assert_called_once()

    def test_history_disabled_for_empty_results(self, mock_retriever, mock_generator, mock_history_storage):
        """Test history behavior when no documents are found."""
        # Mock empty results
        mock_retriever.retrieve.return_value = ([], [])

        disabled_settings = Settings(enable_history=False)

        with patch('local_rag_backend.core.services.rag.settings', disabled_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            # Should NOT save to history even for empty results
            mock_history_storage.save.assert_not_called()
            assert result["answer"] == "No documents are available to answer your question."

    def test_history_enabled_for_empty_results(self, mock_retriever, mock_generator, mock_history_storage):
        """Test history behavior when no documents are found but history is enabled."""
        # Mock empty results
        mock_retriever.retrieve.return_value = ([], [])

        enabled_settings = Settings(enable_history=True, max_history_entries=100)

        with patch('local_rag_backend.core.services.rag.settings', enabled_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            # Should save to history with empty source_ids
            mock_history_storage.save.assert_called_once_with(
                "test question",
                "No documents are available to answer your question.",
                []
            )

    @pytest.mark.parametrize("max_entries,should_cleanup", [
        (0, False),    # Unlimited
        (1, True),     # Very small limit
        (100, True),   # Normal limit
        (10000, True), # Large limit
    ])
    def test_history_cleanup_various_limits(self, mock_retriever, mock_generator, mock_history_storage, max_entries, should_cleanup):
        """Test history cleanup with various limit configurations."""
        settings_config = Settings(max_history_entries=max_entries)

        with patch('local_rag_backend.core.services.rag.settings', settings_config):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
            result = rag_service.ask("test question", top_k=3)

            mock_history_storage.save.assert_called_once()
            if should_cleanup:
                mock_history_storage.cleanup_old_entries.assert_called_once_with(max_entries)
            else:
                mock_history_storage.cleanup_old_entries.assert_not_called()

    def test_history_storage_without_cleanup_capabilities(self, mock_retriever, mock_generator):
        """Test behavior with storage that doesn't support cleanup."""
        basic_storage = Mock()
        basic_storage.save = Mock()
        # No cleanup methods available

        limited_settings = Settings(max_history_entries=100)

        with patch('local_rag_backend.core.services.rag.settings', limited_settings):
            rag_service = RagService(mock_retriever, mock_generator, basic_storage)
            result = rag_service.ask("test question", top_k=3)

            # Should save but not attempt cleanup
            basic_storage.save.assert_called_once()
            assert result["answer"] == "Generated answer"

    def test_history_settings_validation(self):
        """Test that history settings are properly validated."""
        # Valid settings
        valid_settings = [
            Settings(enable_history=True, max_history_entries=0),
            Settings(enable_history=True, max_history_entries=100),
            Settings(enable_history=False, max_history_entries=1000),
        ]

        for settings in valid_settings:
            assert isinstance(settings.enable_history, bool)
            assert isinstance(settings.max_history_entries, int)
            assert settings.max_history_entries >= 0

        # Invalid settings should raise ValueError
        with pytest.raises(ValueError, match="max_history_entries must be non-negative"):
            Settings(max_history_entries=-1)

    def test_history_environment_variable_configuration(self, mock_retriever, mock_generator, mock_history_storage):
        """Test history configuration via environment variables."""
        with patch.dict('os.environ', {
            'ENABLE_HISTORY': 'false',
            'MAX_HISTORY_ENTRIES': '500'
        }):
            env_settings = Settings()

            with patch('local_rag_backend.core.services.rag.settings', env_settings):
                rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)
                result = rag_service.ask("test question", top_k=3)

                # Should respect environment configuration
                mock_history_storage.save.assert_not_called()  # History disabled

    def test_history_multiple_interactions_cleanup(self, mock_retriever, mock_generator, mock_history_storage):
        """Test cleanup behavior across multiple interactions."""
        limited_settings = Settings(max_history_entries=10)

        with patch('local_rag_backend.core.services.rag.settings', limited_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)

            # Make multiple requests
            for i in range(5):
                result = rag_service.ask(f"question {i}", top_k=3)
                assert result["answer"] == "Generated answer"

            # Should have saved 5 times and triggered cleanup 5 times
            assert mock_history_storage.save.call_count == 5
            assert mock_history_storage.cleanup_old_entries.call_count == 5

    def test_history_performance_impact_minimal(self, mock_retriever, mock_generator, mock_history_storage):
        """Test that history management doesn't significantly impact performance."""
        # This is a basic test - in real scenarios you'd measure timing
        limited_settings = Settings(max_history_entries=100)

        with patch('local_rag_backend.core.services.rag.settings', limited_settings):
            rag_service = RagService(mock_retriever, mock_generator, mock_history_storage)

            # Multiple rapid calls should all succeed
            for i in range(10):
                result = rag_service.ask(f"rapid question {i}", top_k=3)
                assert result["answer"] == "Generated answer"

            # All should have been processed
            assert mock_history_storage.save.call_count == 10
