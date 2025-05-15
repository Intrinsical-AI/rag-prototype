# tests/unit/test_generation_ollama.py
import pytest
from unittest import mock
import requests  # Para los tipos de excepciones
from fastapi import HTTPException

from src.adapters.generation.ollama_chat import OllamaGenerator
from src.settings import settings  # Para las configuraciones


@pytest.fixture
def ollama_generator() -> OllamaGenerator:
    # Si necesitas mockear settings para este test, usa monkeypatch
    return OllamaGenerator()


@mock.patch("requests.post")  # Mockea requests.post
def test_ollama_generator_happy_path(mock_post, ollama_generator: OllamaGenerator):
    # Arrange
    question = "Ollama test question?"
    contexts = ["Ollama context 1.", "Ollama context 2."]
    expected_ollama_answer = "This is a mock Ollama answer."

    mock_response_object = mock.MagicMock()
    mock_response_object.json.return_value = {"response": expected_ollama_answer}
    mock_response_object.status_code = 200
    mock_response_object.raise_for_status.return_value = None  # No levanta error
    mock_post.return_value = mock_response_object

    # Act
    answer = ollama_generator.generate(question, contexts)

    # Assert
    assert answer == expected_ollama_answer
    mock_post.assert_called_once()

    args, kwargs = mock_post.call_args
    # print(args)
    # print(kwargs)

    expected_api_url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    assert args[0] == expected_api_url

    payload = kwargs["json"]
    assert payload["model"] == settings.ollama_model
    assert payload["stream"] is False

    ctx_block_expected = "\n".join(f"- {c}" for c in contexts)
    full_prompt_expected = (
        "Based on the following context, please answer the question.\nIf the context does not provide an answer, say so.\n\n"
        "CONTEXT:\n"
        f"{ctx_block_expected}\n\n"
        "QUESTION:\n"
        f"{question}"
    )
    assert payload["prompt"] == full_prompt_expected
    assert kwargs["timeout"] == settings.ollama_request_timeout


@mock.patch("requests.post")
def test_ollama_generator_handles_connection_error(
    mock_post, ollama_generator: OllamaGenerator
):
    # Arrange
    mock_post.side_effect = requests.exceptions.ConnectionError("Failed to connect")

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        ollama_generator.generate("q", ["c"])

    assert exc_info.value.status_code == 503
    assert "Could not connect to Ollama server" in exc_info.value.detail


@mock.patch("requests.post")
def test_ollama_generator_handles_http_error(
    mock_post, ollama_generator: OllamaGenerator
):
    # Arrange
    mock_response_object = mock.MagicMock()
    mock_response_object.status_code = 500
    mock_response_object.text = "Internal Server Error from Ollama"
    mock_response_object.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Mock HTTP Error", response=mock_response_object
    )
    mock_post.return_value = mock_response_object

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        ollama_generator.generate("q", ["c"])

    assert exc_info.value.status_code == 500
    assert (
        "Ollama API error: Internal Server Error from Ollama" in exc_info.value.detail
    )


@mock.patch("requests.post")
def test_ollama_generator_handles_malformed_json_response(
    mock_post, ollama_generator: OllamaGenerator
):
    # Arrange
    mock_response_object = mock.MagicMock()
    mock_response_object.status_code = 200
    mock_response_object.raise_for_status.return_value = None
    mock_response_object.json.return_value = {
        "error": "unexpected_format_no_response_key"
    }  # Falta la clave 'response'
    mock_post.return_value = mock_response_object

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        ollama_generator.generate("q", ["c"])

    assert exc_info.value.status_code == 500
    assert "Ollama response malformed: 'response' key missing" in exc_info.value.detail
