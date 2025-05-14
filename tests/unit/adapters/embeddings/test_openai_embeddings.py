# tests/unit/adapters/embeddings/test_openai_embeddings.py
import pytest
from unittest import mock
from openai import APIError # Importar el error de la API v1.x

from src.adapters.embeddings.openai import OpenAIEmbedder
from src.settings import settings # Para verificar el modelo usado

# No necesitamos la fixture openai_embedder_instance si instanciamos dentro de cada test
# donde el mock de la clase OpenAI está activo.

@mock.patch("src.adapters.embeddings.openai.OpenAI") # Mockea la CLASE OpenAI
def test_openai_embedder_happy_path(MockOpenAI, monkeypatch):
    # Arrange
    if settings.openai_api_key is None: # Asegurar API key para el __init__ del embedder
        monkeypatch.setattr(settings, "openai_api_key", "sk-dummy-for-happy-path")

    text_to_embed = "This is a test sentence for happy path."
    # Crear un vector de la dimensión correcta (1536 para text-embedding-3-small por defecto en settings)
    # Usamos el DIM del embedder que se instancia para asegurar consistencia.
    # La instanciación de OpenAIEmbedder debe ocurrir después de cualquier monkeypatch de settings.openai_embedding_model
    # para que DIM se calcule correctamente. Aquí usamos el DIM por defecto.
    
    # Primero instanciamos el embedder para obtener su DIM configurado
    # (asumiendo que settings.openai_embedding_model no se cambia para este test)
    temp_embedder_for_dim = OpenAIEmbedder()
    expected_dim = temp_embedder_for_dim.DIM
    expected_embedding_vector = [i * 0.001 for i in range(expected_dim)]


    mock_client_instance = MockOpenAI.return_value # La instancia que OpenAI() devolvería

    # Configurar el mock para que devuelva una estructura similar a la de OpenAI Embeddings API v1.x
    mock_embedding_api_response = mock.MagicMock()
    mock_embedding_data_item = mock.MagicMock()
    mock_embedding_data_item.embedding = expected_embedding_vector
    mock_embedding_api_response.data = [mock_embedding_data_item]
    
    mock_client_instance.embeddings.create.return_value = mock_embedding_api_response

    # Instanciar el embedder que realmente probaremos, DESPUÉS de que el mock esté activo
    # y después de cualquier monkeypatch que afecte a su __init__.
    embedder_under_test = OpenAIEmbedder() # Su __init__ llamará a MockOpenAI()

    # Act
    embedding_result = None
    try:
        embedding_result = embedder_under_test.embed(text_to_embed)
        mock_client_instance.embeddings.create.assert_called_once_with(
            model=settings.openai_embedding_model,
            input=text_to_embed
        )
    except Exception as e:
        pytest.fail(f"Embed en happy path falló inesperadamente: {e}")

    # Assert
    assert embedding_result == expected_embedding_vector
    assert embedder_under_test.DIM == expected_dim # Verificar que el DIM es el esperado


@mock.patch("src.adapters.embeddings.openai.OpenAI")
def test_openai_embedder_handles_api_error(MockOpenAI, monkeypatch):
    # Arrange
    if settings.openai_api_key is None:
        monkeypatch.setattr(settings, "openai_api_key", "sk-dummy-for-api-error")

    text_to_embed = "Sentence that will cause API error."
    
    mock_client_instance = MockOpenAI.return_value
    mock_client_instance.embeddings.create.side_effect = APIError(
        message="Mock Embedding API Error from test", request=None, body=None
    )

    # Instanciar el embedder que probaremos
    embedder_under_test = OpenAIEmbedder()

    # Act & Assert
    with pytest.raises(APIError) as exc_info:
        embedder_under_test.embed(text_to_embed)
    
    assert "Mock Embedding API Error from test" in str(exc_info.value)
    mock_client_instance.embeddings.create.assert_called_once_with(
        model=settings.openai_embedding_model, # Verificar que se intentó llamar con los params correctos
        input=text_to_embed
    )

@mock.patch("src.adapters.embeddings.openai.OpenAI")
def test_openai_embedder_dim_updates_with_model_setting(MockOpenAI, monkeypatch):
    # Arrange
    original_embedding_model = settings.openai_embedding_model # Guardar para restaurar
    
    # Caso 1: text-embedding-3-large
    monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-3-large")
    if settings.openai_api_key is None:
        monkeypatch.setattr(settings, "openai_api_key", "sk-dummy-for-dim-test-large")
    
    # MockOpenAI se aplica aquí, así que la instancia de OpenAI() será un mock
    embedder_large = OpenAIEmbedder() 
    assert embedder_large.DIM == 3072, "DIM should be 3072 for text-embedding-3-large"
    # No necesitamos verificar la llamada a .create() aquí, solo el __init__ y DIM.
    # MockOpenAI().assert_called_once() # Verifica que el constructor de OpenAI (mockeado) fue llamado.
                                      # Esto es un poco más avanzado, puede que necesites
                                      # mock_constructor = MockOpenAI; mock_constructor.assert_called_once()
                                      # o verificar que mock_client_instance fue "creado"
    MockOpenAI.assert_called_with(api_key=settings.openai_api_key) # Verificar args de OpenAI()

    # Limpiar el mock de llamada para el siguiente caso (si MockOpenAI fuera persistente entre llamadas)
    # pero como se re-instancia el embedder, MockOpenAI() se llama de nuevo.
    MockOpenAI.reset_mock() # Resetea conteos de llamadas, etc.

    # Caso 2: text-embedding-ada-002 (también tiene DIM 1536, pero probamos el setting)
    monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-ada-002")
    if settings.openai_api_key is None: # Redundante si ya se seteó, pero no daña
        monkeypatch.setattr(settings, "openai_api_key", "sk-dummy-for-dim-test-ada")

    embedder_ada = OpenAIEmbedder()
    assert embedder_ada.DIM == 1536, "DIM should be 1536 for text-embedding-ada-002"
    MockOpenAI.assert_called_with(api_key=settings.openai_api_key)

    # Restaurar el setting original para no afectar otros tests si se ejecutan en la misma sesión
    # (aunque pytest debería aislar, es buena práctica con monkeypatch a nivel de settings globales)
    monkeypatch.setattr(settings, "openai_embedding_model", original_embedding_model)