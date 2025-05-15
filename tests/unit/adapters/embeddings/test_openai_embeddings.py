# tests/unit/adapters/embeddings/test_openai_embeddings.py
import pytest
from unittest import mock
from openai import APIError # type: ignore

from src.adapters.embeddings.openai import OpenAIEmbedder
from src.settings import settings


@mock.patch("src.adapters.embeddings.openai.OpenAI")
def test_openai_embedder_happy_path(MockOpenAI, monkeypatch):
    """Test embedding with a successful OpenAI API response."""
    # Arrange
    # Ensure API key is set for the test if not already by conftest
    current_api_key = settings.openai_api_key
    if not current_api_key:
        monkeypatch.setattr(settings, "openai_api_key", "sk-test-key-happy-path")
    
    text = "This is a test sentence for embedding."

    # Instantiate embedder ONCE
    embedder = OpenAIEmbedder() # This will call MockOpenAI constructor
    expected_dim = embedder.DIM
    expected_vector = [float(i) * 0.001 for i in range(expected_dim)]

    mock_openai_instance = MockOpenAI.return_value # Get the mocked instance
    mock_embedding_response = mock.MagicMock()
    mock_embedding_data_item = mock.MagicMock()
    mock_embedding_data_item.embedding = expected_vector
    mock_embedding_response.data = [mock_embedding_data_item]

    mock_openai_instance.embeddings.create.return_value = mock_embedding_response

    # Act
    result = embedder.embed(text) # Use the same embedder instance

    # Assert
    # The OpenAI constructor should have been called once during embedder instantiation
    MockOpenAI.assert_called_once_with(api_key=settings.openai_api_key)
    
    mock_openai_instance.embeddings.create.assert_called_once_with(
        model=settings.openai_embedding_model,
        input=text
    )
    assert result == expected_vector
    assert len(result) == expected_dim

    # Restore original API key if it was patched specifically for this test
    if not current_api_key and settings.openai_api_key == "sk-test-key-happy-path":
        monkeypatch.setattr(settings, "openai_api_key", None)


@mock.patch("src.adapters.embeddings.openai.OpenAI")
def test_openai_embedder_handles_api_error(MockOpenAI, monkeypatch):
    """Test embedder raises exception when OpenAI API returns an error."""
    # Arrange
    current_api_key = settings.openai_api_key
    if not current_api_key:
        monkeypatch.setattr(settings, "openai_api_key", "sk-test-key-api-error")

    text = "This sentence will trigger an API error."

    mock_openai_instance = MockOpenAI.return_value
    mock_openai_instance.embeddings.create.side_effect = APIError(
        message="Mock Embedding API Error",
        request=None, # type: ignore
        body=None
    )

    embedder = OpenAIEmbedder() # Instantiate once

    # Act & Assert
    with pytest.raises(APIError) as exc_info:
        embedder.embed(text)

    assert "Mock Embedding API Error" in str(exc_info.value)
    mock_openai_instance.embeddings.create.assert_called_once_with(
        model=settings.openai_embedding_model,
        input=text
    )
    MockOpenAI.assert_called_once_with(api_key=settings.openai_api_key) # Constructor called once

    if not current_api_key and settings.openai_api_key == "sk-test-key-api-error":
        monkeypatch.setattr(settings, "openai_api_key", None)


@mock.patch("src.adapters.embeddings.openai.OpenAI")
def test_openai_embedder_dim_updates_with_model_setting(MockOpenAI, monkeypatch):
    """Test that embedder dimension correctly updates based on configured model."""
    original_model = settings.openai_embedding_model
    current_api_key = settings.openai_api_key
    if not current_api_key:
        monkeypatch.setattr(settings, "openai_api_key", "sk-test-key-dim-update")

    try:
        # Test case: text-embedding-3-large (expected DIM=3072)
        monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-3-large")
        
        embedder_large = OpenAIEmbedder()
        assert embedder_large.DIM == 3072
        MockOpenAI.assert_called_with(api_key=settings.openai_api_key) # Called for embedder_large

        MockOpenAI.reset_mock() # Reset mock for the next instantiation

        # Test case: text-embedding-ada-002 (expected DIM=1536)
        monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-ada-002")
        embedder_ada = OpenAIEmbedder()
        assert embedder_ada.DIM == 1536
        MockOpenAI.assert_called_with(api_key=settings.openai_api_key) # Called for embedder_ada

    finally:
        # Restore original model setting and API key
        monkeypatch.setattr(settings, "openai_embedding_model", original_model)
        if not current_api_key and settings.openai_api_key == "sk-test-key-dim-update":
            monkeypatch.setattr(settings, "openai_api_key", None)