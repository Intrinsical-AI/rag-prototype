# tests/integration/test_api_ask.py
import pytest
from fastapi.testclient import TestClient
from unittest import mock
import random
import importlib # Para recargar módulos si es necesario
from src.settings import settings
from src.app.main import app # app se importa después de conftest.py
from src.app import dependencies as app_dependencies_module # Para re-init

# --- Fixture para el TestClient (ya no necesita ser module-scoped si la app es estable) ---
@pytest.fixture(scope="function")
def client(populated_db_for_integration) -> TestClient: # Depende de la DB poblada
    # populated_db_for_integration se ejecuta primero, asegura que la DB tiene datos.
    # reset_rag_service_singleton (si es autouse) también se habrá ejecutado.
    from src.app import dependencies as app_dependencies # Para resetear el singleton si es necesario
    
    # Asegurar que el servicio se reinicializa para ESTA instancia de TestClient,
    # especialmente si los settings fueron monkeypatcheados por un test anterior.
    app_dependencies._rag_service = None 
    importlib.reload(app_dependencies) # Recarga para asegurar que init_rag_service usa los settings actuales

    # Cuando TestClient(app) se crea, el lifespan de la app se ejecuta,
    # llamando a init_rag_service. Esta llamada ahora encontrará la DB poblada.
    with TestClient(app) as c:
        yield c

# --- Fixture para limpiar la tabla de historial ---
@pytest.fixture(scope="function")
def clean_history_table():
    from src.db.base import SessionLocal # Usar el SessionLocal ya reconfigurado por conftest
    from src.db.models import QaHistory
    with SessionLocal() as db:
        db.query(QaHistory).delete()
        db.commit()
    yield

# --- Helper para mock de OpenAI API v1.x ---
def _make_openai_v1_mock(answer_text: str):
    mock_completion = mock.MagicMock()
    mock_completion.choices = [mock.MagicMock()]
    mock_completion.choices[0].message = mock.MagicMock()
    mock_completion.choices[0].message.content = answer_text
    return mock_completion

# --- Tests ---
@mock.patch("src.adapters.generation.openai_chat.OpenAI") # Path donde OpenAI es instanciado
def test_api_ask_with_openai_retrieves_and_generates(
    MockOpenAIClass,
    client: TestClient,
    monkeypatch
):
    # Arrange
    # 1. Asegurar que se usará OpenAIGenerator y que tiene una API key para __init__
    monkeypatch.setattr(settings, "ollama_enabled", False)
    current_openai_api_key = settings.openai_api_key # Guardar por si se parchea
    if settings.openai_api_key is None or settings.openai_api_key == "sk-dummy-test-key": # Evitar sobreescribir una real
        monkeypatch.setattr(settings, "openai_api_key", "sk-integration-openai-ask")

    # 2. Configurar el mock para la clase OpenAI
    mock_openai_client_instance = MockOpenAIClass.return_value
    expected_answer = "Mocked AI answer about our refund policy."
    mock_openai_client_instance.chat.completions.create.return_value = _make_openai_v1_mock(expected_answer)

    # 3. Forzar la re-inicialización de RagService para que use el OpenAIGenerator
    #    que a su vez usará la clase OpenAI mockeada.
    #    Y para que el retriever se cree con el modo correcto si lo cambiamos.
    monkeypatch.setattr(settings, "retrieval_mode", "sparse") # Asegurar modo para aserción de source_id
    importlib.reload(app_dependencies_module) # Recargar para que vea los settings parcheados
    app_dependencies_module.init_rag_service() # Recrea _rag_service

    question_text = "What is the refund policy?"
    payload = {"question": question_text}

    # Act
    response = client.post("/api/ask", json=payload)

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == expected_answer
    assert isinstance(body["source_ids"], list)
    assert 1 in body["source_ids"] # Basado en datos de prueba en conftest.py

    mock_openai_client_instance.chat.completions.create.assert_called_once()
    args, kwargs = mock_openai_client_instance.chat.completions.create.call_args
    sent_prompt_content = kwargs["messages"][0]["content"]
    assert question_text in sent_prompt_content
    assert "refund policy states" in sent_prompt_content.lower() # Contexto del Doc ID 1

    # Restaurar API key si se cambió, aunque monkeypatch debería hacerlo por test
    monkeypatch.setattr(settings, "openai_api_key", current_openai_api_key)


@mock.patch("requests.post") # Path donde requests.post es llamado por OllamaGenerator
def test_api_ask_with_ollama_retrieves_and_generates(
    mock_requests_post,
    client: TestClient,
    monkeypatch
):
    # Arrange
    # 1. Habilitar Ollama y deshabilitar retrieval denso para simplificar
    monkeypatch.setattr(settings, "ollama_enabled", True)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse") # O el modo que quieras probar

    # 2. Configurar mock para requests.post (Ollama)
    expected_answer = "Mocked Ollama answer regarding support."
    mock_ollama_response_obj = mock.MagicMock()
    mock_ollama_response_obj.json.return_value = {"response": expected_answer}
    mock_ollama_response_obj.status_code = 200
    mock_ollama_response_obj.raise_for_status.return_value = None
    mock_requests_post.return_value = mock_ollama_response_obj

    # 3. Forzar la re-inicialización de RagService para que use OllamaGenerator
    #    y el retriever correcto.
    importlib.reload(app_dependencies_module)
    app_dependencies_module.init_rag_service()

    question_text = "How to contact support?"
    payload = {"question": question_text}

    # Act
    response = client.post("/api/ask", json=payload)

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == expected_answer
    assert 2 in body["source_ids"] # Basado en datos de prueba en conftest.py

    mock_requests_post.assert_called_once()
    args, kwargs = mock_requests_post.call_args
    sent_payload = kwargs["json"]
    assert question_text in sent_payload["prompt"]
    assert "contact support, please email" in sent_payload["prompt"].lower() # Contexto del Doc ID 2


def test_api_ask_validation_error_missing_question(client: TestClient):
    response = client.post("/api/ask", json={})
    assert response.status_code == 422
    detail = response.json().get("detail", [])
    assert any("question" in error.get("loc", []) for error in detail if isinstance(error, dict) and "loc" in error)


def test_api_ask_validation_error_wrong_type(client: TestClient):
    response = client.post("/api/ask", json={"question": 123})
    assert response.status_code == 422
    detail = response.json().get("detail", [])
    assert any("Input should be a valid string" in error.get("msg", "") for error in detail if isinstance(error, dict) and "msg" in error)


@mock.patch("src.adapters.generation.openai_chat.OpenAI")
def test_history_endpoint_records_and_retrieves_qa(
    MockOpenAIClass,
    client: TestClient,
    monkeypatch,
    clean_history_table # Usar fixture para limpiar tabla
):
    # Arrange
    # 1. Configurar para usar OpenAIGenerator con mock
    monkeypatch.setattr(settings, "ollama_enabled", False)
    current_openai_api_key = settings.openai_api_key
    if settings.openai_api_key is None or settings.openai_api_key == "sk-dummy-test-key":
        monkeypatch.setattr(settings, "openai_api_key", "sk-integration-hist-test")
    
    mock_openai_client_instance = MockOpenAIClass.return_value

    # Forzar re-init de RagService para asegurar que usa estos settings
    # (aunque conftest.py ya lo hace una vez, ollama_enabled podría haber cambiado)
    importlib.reload(app_dependencies_module)
    app_dependencies_module.init_rag_service()

    # Primera Q&A
    q_text1 = f"History Q1 {random.randint(1000, 9999)}"
    ans_text1 = "Hist Ans 1"
    mock_openai_client_instance.chat.completions.create.return_value = _make_openai_v1_mock(ans_text1)
    resp1 = client.post("/api/ask", json={"question": q_text1})
    assert resp1.status_code == 200

    # Segunda Q&A
    q_text2 = f"History Q2 {random.randint(1000, 9999)}"
    ans_text2 = "Hist Ans 2"
    # Importante: .create es un mock, si la misma instancia de mock_openai_client_instance se usa,
    # su return_value para .create necesita ser reconfigurado si la respuesta cambia.
    mock_openai_client_instance.chat.completions.create.return_value = _make_openai_v1_mock(ans_text2)
    resp2 = client.post("/api/ask", json={"question": q_text2})
    assert resp2.status_code == 200
    
    # Act
    resp_hist = client.get("/api/history?limit=5")
    assert resp_hist.status_code == 200
    history_data = resp_hist.json()

    # Assert
    assert len(history_data) == 2 # Exactamente 2 debido a clean_history_table

    assert history_data[0]["question"] == q_text2 # Más reciente primero
    assert history_data[0]["answer"] == ans_text2
    assert history_data[1]["question"] == q_text1
    assert history_data[1]["answer"] == ans_text1

    assert mock_openai_client_instance.chat.completions.create.call_count == 2

    monkeypatch.setattr(settings, "openai_api_key", current_openai_api_key)