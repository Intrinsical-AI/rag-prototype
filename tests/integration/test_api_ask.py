# tests/integration/test_api_ask.py

import pytest
from fastapi.testclient import TestClient
from unittest import mock
import random
import importlib

from src.settings import settings
from src.app.main import app  # arranca la app con Lifespan
from src.app import dependencies as app_dependencies_module

# --- Fixture para TestClient limpio en cada test ---
@pytest.fixture(scope="function")
def client(populated_db_for_integration) -> TestClient:
    # Resetear singleton del RagService entre tests
    app_dependencies_module._rag_service = None
    importlib.reload(app_dependencies_module)

    with TestClient(app) as c:
        yield c

# --- Fixture para limpiar historial ---
@pytest.fixture(scope="function")
def clean_history_table():
    from src.db.base import SessionLocal
    from src.db.models import QaHistory

    with SessionLocal() as db:
        db.query(QaHistory).delete()
        db.commit()
    yield

# --- Helper para mock de OpenAI API v1.x ---
def _make_openai_v1_mock(answer_text: str):
    mock_completion = mock.MagicMock()
    mock_choice = mock.MagicMock()
    mock_choice.message.content = answer_text
    mock_completion.choices = [mock_choice]
    return mock_completion

# --- Tests ---

@mock.patch("src.adapters.generation.openai_chat.OpenAI")
def test_api_ask_with_openai_retrieves_and_generates(
    MockOpenAIClass,
    client: TestClient,
    monkeypatch
):
    # Forzar uso de OpenAI
    monkeypatch.setattr(settings, "ollama_enabled", False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")
    # Asegurar key de test
    monkeypatch.setattr(settings, "openai_api_key", "sk-integration-openai-ask")

    # Preparar mock de OpenAI
    mock_openai = MockOpenAIClass.return_value
    expected = "Mocked AI answer about our refund policy."
    mock_openai.chat.completions.create.return_value = _make_openai_v1_mock(expected)

    # Reinicializar RagService con los settings parcheados
    importlib.reload(app_dependencies_module)
    app_dependencies_module.init_rag_service()

    # Ejecución
    q = "What is the refund policy?"
    resp = client.post("/api/ask", json={"question": q})
    assert resp.status_code == 200

    body = resp.json()
    assert body["answer"] == expected
    assert isinstance(body["source_ids"], list)
    assert 101 in body["source_ids"]

    # Comprobar prompt enviado
    mock_openai.chat.completions.create.assert_called_once()
    _, kwargs = mock_openai.chat.completions.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert q in prompt
    assert "refund policy states" in prompt.lower()


@mock.patch("requests.post")
@mock.patch("requests.get")
def test_api_ask_with_ollama_retrieves_and_generates(
    mock_requests_get,
    mock_requests_post,
    client: TestClient,
    monkeypatch
):
    # Forzar uso de Ollama
    monkeypatch.setattr(settings, "ollama_enabled", True)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")

    # Mock health-check de Ollama (requests.get)
    mock_health = mock.MagicMock()
    mock_health.status_code = 200
    mock_requests_get.return_value = mock_health

    # Mock generación de Ollama (requests.post)
    expected = "Mocked Ollama answer regarding support."
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": expected}
    mock_requests_post.return_value = mock_resp

    # Reinicializar RagService
    importlib.reload(app_dependencies_module)
    app_dependencies_module.init_rag_service()

    # Ejecución
    q = "How to contact support?"
    resp = client.post("/api/ask", json={"question": q})
    assert resp.status_code == 200

    body = resp.json()
    assert body["answer"] == expected
    assert 102 in body["source_ids"]

    # Comprobar llamadas a health-check y generación
    mock_requests_get.assert_called_once()
    mock_requests_post.assert_called_once()
    _, kwargs = mock_requests_post.call_args
    payload = kwargs["json"]
    assert q in payload["prompt"]
    assert "contact support, please email" in payload["prompt"].lower()


def test_api_ask_validation_error_missing_question(client: TestClient):
    resp = client.post("/api/ask", json={})
    assert resp.status_code == 422
    errors = resp.json()["detail"]
    assert any("question" in err.get("loc", []) for err in errors)


def test_api_ask_validation_error_wrong_type(client: TestClient):
    resp = client.post("/api/ask", json={"question": 123})
    assert resp.status_code == 422
    errors = resp.json()["detail"]
    assert any("string" in err.get("msg", "") for err in errors)

@mock.patch("src.adapters.generation.openai_chat.OpenAI")
def test_history_endpoint_records_and_retrieves_qa(
    MockOpenAIClass,
    client: TestClient,
    monkeypatch,
    clean_history_table
):
    # Forzar OpenAI
    monkeypatch.setattr(settings, "ollama_enabled", False)
    monkeypatch.setattr(settings, "openai_api_key", "sk-integration-hist-test")
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")

    mock_openai = MockOpenAIClass.return_value
    # Reinicializar RagService
    importlib.reload(app_dependencies_module)
    app_dependencies_module.init_rag_service()

    # Primera Q&A
    q1 = f"History Q1 {random.randint(1000,9999)}"
    a1 = "Hist Ans 1"
    mock_openai.chat.completions.create.return_value = _make_openai_v1_mock(a1)
    r1 = client.post("/api/ask", json={"question": q1})
    assert r1.status_code == 200

    # Segunda Q&A
    q2 = f"History Q2 {random.randint(1000,9999)}"
    a2 = "Hist Ans 2"
    mock_openai.chat.completions.create.return_value = _make_openai_v1_mock(a2)
    r2 = client.post("/api/ask", json={"question": q2})
    assert r2.status_code == 200

    # Recuperar historial
    rh = client.get("/api/history?limit=5")
    assert rh.status_code == 200
    hist = rh.json()

    assert len(hist) == 2
    assert hist[0]["question"] == q2 and hist[0]["answer"] == a2
    assert hist[1]["question"] == q1 and hist[1]["answer"] == a1

    assert mock_openai.chat.completions.create.call_count == 2
