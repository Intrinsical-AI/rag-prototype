# tests/unit/adapters/embeddings/test_sentence_transformer_embedder.py

from unittest import mock
import numpy as np

from src.adapters.embeddings.sentence_transformers import SentenceTransformerEmbedder


@mock.patch("src.adapters.embeddings.sentence_transformers.SentenceTransformer")
def test_sentence_transformer_embedder_happy_path(MockSentenceTransformer):
    """Test embedding generation with default model successfully."""

    # Arrange
    default_model_name = "all-MiniLM-L6-v2"
    input_text = "This is a test sentence for sentence-transformer."

    # Setup mock for SentenceTransformer instance and encode method
    mock_model_instance = MockSentenceTransformer.return_value

    expected_embedding_array = np.array(
        [i * 0.01 for i in range(384)], dtype=np.float32
    )
    expected_embedding_list = expected_embedding_array.tolist()

    mock_model_instance.encode.return_value = [expected_embedding_array]

    # Act
    embedder = SentenceTransformerEmbedder(model_name=default_model_name)
    embedding_result = embedder.embed(input_text)

    # Assert
    MockSentenceTransformer.assert_called_once_with(default_model_name)
    mock_model_instance.encode.assert_called_once_with([input_text])

    assert embedding_result == expected_embedding_list
    assert isinstance(embedding_result, list)
    assert len(embedding_result) == 384
    assert all(isinstance(x, float) for x in embedding_result)

    assert embedder.DIM == 384


@mock.patch("src.adapters.embeddings.sentence_transformers.SentenceTransformer")
def test_sentence_transformer_embedder_custom_model(MockSentenceTransformer):
    """Test embedding generation with a custom model name successfully."""

    # Arrange
    custom_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    input_text = "Another test sentence."

    mock_model_instance = MockSentenceTransformer.return_value

    expected_embedding_array = np.array(
        [i * 0.02 for i in range(384)], dtype=np.float32
    )
    expected_embedding_list = expected_embedding_array.tolist()

    mock_model_instance.encode.return_value = [expected_embedding_array]

    # Act
    embedder = SentenceTransformerEmbedder(model_name=custom_model_name)
    embedding_result = embedder.embed(input_text)

    # Assert
    MockSentenceTransformer.assert_called_once_with(custom_model_name)
    mock_model_instance.encode.assert_called_once_with([input_text])

    assert embedding_result == expected_embedding_list
    assert isinstance(embedding_result, list)
    assert len(embedding_result) == 384
    assert all(isinstance(x, float) for x in embedding_result)

    # Note: The dimension is currently hardcoded; update this test if DIM becomes dynamic.
    assert embedder.DIM == 384
