# tests/unit/test_generation_openai.py

import pytest
from unittest import mock
from fastapi import HTTPException
from openai import APIError

from src.adapters.generation.openai_chat import OpenAIGenerator
from src.settings import settings


@pytest.fixture
def openai_generator_instance(monkeypatch):
    """
    Fixture that creates an OpenAIGenerator with a dummy API key set via environment variable.
    Cleans up the env var after use.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-test-key-for-env")
    generator = OpenAIGenerator()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return generator


@mock.patch("src.adapters.generation.openai_chat.OpenAI")
def test_openai_generator_happy_path(MockOpenAI):
    """
    Test OpenAIGenerator.generate() returns the expected answer and
    calls the API with the correct arguments.
    """
    question = "Test question?"
    contexts = ["Context 1.", "Context 2 snippet."]
    expected_generated_answer = "This is a mock AI answer for API v1."

    # Mock OpenAI client and completion API
    mock_client_instance = MockOpenAI.return_value
    mock_completion = mock.MagicMock()
    mock_completion.choices = [mock.MagicMock()]
    mock_completion.choices[0].message = mock.MagicMock()
    mock_completion.choices[0].message.content = expected_generated_answer

    mock_client_instance.chat.completions.create.return_value = mock_completion

    # Instantiate generator with the mocked OpenAI class
    generator = OpenAIGenerator()

    # Act
    answer = generator.generate(question, contexts)

    # Assert: output
    assert answer == expected_generated_answer

    # Assert: OpenAI API was called once, with correct parameters
    mock_client_instance.chat.completions.create.assert_called_once()
    _, kwargs = mock_client_instance.chat.completions.create.call_args

    assert kwargs["model"] == settings.openai_model
    assert kwargs["temperature"] == settings.openai_temperature

    # Prompt construction check
    messages = kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    ctx_block_expected = "\n".join(f"- {c}" for c in contexts)
    prompt_content_expected = (
        "Answer using ONLY the context provided.\n\n"
        f"CONTEXT:\n{ctx_block_expected}\n\n"
        f"QUESTION: {question}"
    )
    assert messages[0]["content"] == prompt_content_expected


@mock.patch("src.adapters.generation.openai_chat.OpenAI")
def test_openai_generator_handles_api_error(MockOpenAI):
    """
    Test OpenAIGenerator.generate() raises HTTPException with correct code
    and error message if OpenAI APIError is raised.
    """
    question = "Another question"
    contexts = ["Some context."]

    # Configure the mock to raise APIError on completion call
    mock_client_instance = MockOpenAI.return_value
    mock_client_instance.chat.completions.create.side_effect = APIError(
        message="Mock API Error from v1", request=None, body=None
    )

    generator = OpenAIGenerator()

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        generator.generate(question, contexts)

    assert exc_info.value.status_code == 502
    assert "OpenAI API Error: Mock API Error from v1" in str(exc_info.value.detail)
    mock_client_instance.chat.completions.create.assert_called_once()
