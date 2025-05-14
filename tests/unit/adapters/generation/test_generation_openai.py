# tests/unit/test_generation_openai.py
import pytest
from unittest import mock
from fastapi import HTTPException
from openai import APIError # Importar el nuevo error

from src.adapters.generation.openai_chat import OpenAIGenerator
from src.settings import settings

@pytest.fixture
def openai_generator_instance(monkeypatch) -> OpenAIGenerator:
    # Si settings.openai_api_key es None y confías en la var de entorno
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-test-key-for-env")
    generator = OpenAIGenerator()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False) # Limpiar después
    return generator

# El path para mockear cambia a donde se encuentra el método que queremos interceptar.
# Como 'self.client' se crea en el __init__ de OpenAIGenerator, y luego se llama
# 'self.client.chat.completions.create', necesitamos mockear ese método.
# Esto se puede hacer mockeando la clase OpenAI y su cadena de atributos.
@mock.patch("src.adapters.generation.openai_chat.OpenAI") # Mockea la clase OpenAI donde se importa
def test_openai_generator_happy_path(MockOpenAI, openai_generator_instance: OpenAIGenerator):
    # Arrange
    question = "Test question?"
    contexts = ["Context 1.", "Context 2 snippet."]
    expected_generated_answer = "This is a mock AI answer for API v1."

    # Configurar el mock de la instancia del cliente y su método
    mock_client_instance = MockOpenAI.return_value # Esto es lo que self.client será
    mock_completion = mock.MagicMock()
    mock_completion.choices = [mock.MagicMock()]
    mock_completion.choices[0].message = mock.MagicMock()
    mock_completion.choices[0].message.content = expected_generated_answer
    
    # El método mockeado ahora es en la instancia del cliente
    mock_client_instance.chat.completions.create.return_value = mock_completion

    # Act
    # Necesitamos re-instanciar OpenAIGenerator DESPUÉS de que MockOpenAI esté activo
    # para que su __init__ use el cliente mockeado, O inyectar el cliente mockeado.
    # La forma más simple es que la fixture ya use el mock si es posible, o
    # que el generador se cree aquí.
    
    # Si OpenAIGenerator crea self.client en __init__, y la fixture lo crea antes del mock,
    # el cliente no será el mockeado.
    # Opción 1: Recrear el generador aquí (más simple para este test)
    generator_under_test = OpenAIGenerator() # Su __init__ ahora usará MockOpenAI().return_value

    answer = generator_under_test.generate(question, contexts)

    # Assert
    assert answer == expected_generated_answer

    # Verificar que el método create fue llamado una vez en la instancia mockeada del cliente
    mock_client_instance.chat.completions.create.assert_called_once()
    
    args, kwargs = mock_client_instance.chat.completions.create.call_args
    # print(kwargs) # Descomenta para ver los kwargs

    assert kwargs["model"] == settings.openai_model
    # La API Key ya no se pasa a 'create', se usa al instanciar el cliente OpenAI.
    # assert kwargs["api_key"] == settings.openai_api_key # YA NO ES ASÍ
    assert kwargs["temperature"] == settings.openai_temperature

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
def test_openai_generator_handles_api_error(MockOpenAI, openai_generator_instance: OpenAIGenerator): # La fixture no se usa directamente aquí
    # Arrange
    question = "Another question"
    contexts = ["Some context."]
    
    mock_client_instance = MockOpenAI.return_value
    # Configurar el mock para que levante un APIError de OpenAI v1.x
    # El error puede tener un 'response' mockeado si tu código lo usa.
    # Para un error simple, un mensaje es suficiente.
    # Necesitas crear un objeto mock de respuesta si tu manejo de errores lo espera.
    # Por ahora, un APIError simple.
    mock_client_instance.chat.completions.create.side_effect = APIError(
        message="Mock API Error from v1", 
        request=None, # puedes mockear request si es necesario
        body=None     # puedes mockear body si es necesario
    )
    # Si necesitas simular un status_code específico, APIError lo puede tomar o se puede
    # construir un mock_response para el error.
    # error_response = mock.MagicMock()
    # error_response.status_code = 429 # Ejemplo
    # mock_client_instance.chat.completions.create.side_effect = APIError(
    #     message="Rate limit exceeded", request=None, body=None, response=error_response
    # )


    generator_under_test = OpenAIGenerator() # Para que use el cliente mockeado

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        generator_under_test.generate(question, contexts)
    
    assert exc_info.value.status_code == 502 # O el código que decidas para el error
    # El mensaje de error ahora vendrá de err.message del APIError
    assert "OpenAI API Error: Mock API Error from v1" in str(exc_info.value.detail)

    mock_client_instance.chat.completions.create.assert_called_once()