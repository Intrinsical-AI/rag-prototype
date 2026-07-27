# tests/unit/app/test_api_router_more.py

import pytest

from local_rag_backend.composition import factory
from local_rag_backend.core.errors import LLMConfigurationError, LLMConnectionError, LLMTimeoutError
from local_rag_backend.http import dependencies as deps
from local_rag_backend.http.routers import health as health_router, rag_router
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_health_endpoint_ok(asgi_client):
    r = await asgi_client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


async def test_ready_endpoint_ok(asgi_client, monkeypatch):
    class _Dummy:
        pass

    # Ensure at least one provider is available
    monkeypatch.setattr(settings, "openai_api_key", "x", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    async def _override():
        return _Dummy()

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)
    r = await asgi_client.get("/readyz")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert "database" in data["checks"]
    assert "rag_service" in data["checks"]
    assert "llm_providers" in data["checks"]


async def test_ready_endpoint_not_ready_no_llm(asgi_client, monkeypatch):
    class _Dummy:
        pass

    # No providers configured
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    async def _override():
        return _Dummy()

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)
    r = await asgi_client.get("/readyz")
    assert r.status_code == 503
    payload = r.json()
    assert payload["status"] == "not_ready"
    assert "llm_providers" in payload["checks"]


async def test_templates_endpoint(asgi_client):
    r = await asgi_client.get("/api/templates")
    assert r.status_code == 200
    arr = r.json()
    assert isinstance(arr, list) and len(arr) >= 3
    names = {t["name"] for t in arr}
    assert {"default", "ollama"}.issubset(names)


async def test_config_endpoint_providers(asgi_client, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "key", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", True, raising=False)
    monkeypatch.setattr(settings, "openrouter_enabled", True, raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "k", raising=False)

    r = await asgi_client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    prov = set(data.get("available_providers", []))
    assert {"openai", "ollama", "openrouter"}.issubset(prov)
    assert data["retrieval_mode"] in {"sparse", "dense", "hybrid"}


async def test_ask_eval_invalid_config(asgi_client, monkeypatch):
    payload = {
        "question": "q",
        "config": {
            "retrieval_mode": "invalid-mode",
            "k": 3,
        },
    }
    r = await asgi_client.post("/api/ask_eval", json=payload)
    assert r.status_code == 400
    assert "Invalid config" in r.json()["detail"]


@pytest.mark.parametrize(
    "config,expected_status",
    [
        ({"retrieval_mode": "invalid", "k": 3}, 400),  # invalid mode (runtime validation)
        ({"retrieval_mode": "sparse", "k": 0}, 422),  # k too small (schema)
        ({"retrieval_mode": "dense", "k": 11}, 422),  # k too large (schema)
        ({"retrieval_mode": "hybrid", "k": 3, "hybrid_alpha": -0.1}, 422),  # schema
        ({"retrieval_mode": "hybrid", "k": 3, "hybrid_alpha": 1.1}, 422),  # schema
        ({"retrieval_mode": "sparse", "k": 3, "temperature": -0.5}, 422),  # schema
        ({"retrieval_mode": "sparse", "k": 3, "temperature": 2.5}, 422),  # schema
        ({"retrieval_mode": "sparse", "k": 3, "top_p": -0.1}, 422),  # schema
        ({"retrieval_mode": "sparse", "k": 3, "top_p": 1.1}, 422),  # schema
        ({"retrieval_mode": "sparse", "k": 3, "max_tokens": 0}, 422),  # schema
        ({"retrieval_mode": "sparse", "k": 3, "max_tokens": 999999}, 422),  # schema
    ],
)
async def test_ask_eval_invalid_config_parametrized_async(asgi_client, config, expected_status):
    payload = {"question": "q", "config": config}
    r = await asgi_client.post("/api/ask_eval", json=payload)
    assert r.status_code == expected_status
    if expected_status == 400:
        assert "Invalid config" in r.json().get("detail", "")


async def test_openrouter_generate_not_configured(asgi_client, monkeypatch):
    monkeypatch.setattr(settings, "openrouter_enabled", False, raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", None, raising=False)
    r = await asgi_client.post(
        "/api/openrouter/generate",
        json={
            "model": None,
            "system_instruction": "sys",
            "user_content": "hi",
        },
    )
    assert r.status_code == 400


async def test_dependencies_no_llm(monkeypatch, in_memory_sqlite, reset_app_context):
    _ = reset_app_context
    deps.reset_rag_service()
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    with pytest.raises(LLMConfigurationError):
        await deps.get_rag_service()


async def test_ask_returns_503_without_llm_for_valid_payload(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)
    monkeypatch.setattr(settings, "openrouter_enabled", False, raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", None, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    r = await asgi_client.post("/api/ask", json={"question": "hi", "k": 1})
    assert r.status_code == 503
    assert r.json()["detail"] == "Service unavailable."
    assert "No LLM configured" not in r.text


@pytest.mark.parametrize("payload", [{"question": "", "k": 1}, {"question": "hi", "k": 0}])
async def test_ask_validation_errors_take_precedence_without_llm(
    asgi_client, in_memory_sqlite, monkeypatch, payload
):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)
    monkeypatch.setattr(settings, "openrouter_enabled", False, raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", None, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    r = await asgi_client.post("/api/ask", json=payload)
    assert r.status_code == 422


async def test_metrics_endpoint_disabled(asgi_client, monkeypatch):
    monkeypatch.setattr(settings, "enable_monitoring", False, raising=False)
    r = await asgi_client.get("/metrics")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/plain")
    assert r.text.startswith("# Monitoring disabled")


async def test_ask_maps_typed_llm_timeout_to_504(asgi_client, in_memory_sqlite, monkeypatch):
    class _FailingService:
        def ask(self, question, top_k=3):
            raise LLMTimeoutError("provider timeout")

    async def _override():
        return _FailingService()

    monkeypatch.setattr(rag_router, "get_rag_service", _override, raising=True)
    r = await asgi_client.post("/api/ask", json={"question": "hi", "k": 1})

    assert r.status_code == 504
    assert r.json()["detail"] == "Gateway timeout."
    assert "provider timeout" not in r.text


async def test_ask_eval_maps_typed_llm_connection_error_to_503(
    asgi_client, in_memory_sqlite, monkeypatch
):
    SqlDocumentStorage().store_documents(["hello world"])

    class _FailingGenerator:
        def generate(self, question, contexts):
            raise LLMConnectionError("provider unreachable")

    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIGenerator", lambda **_k: _FailingGenerator(), raising=True)

    r = await asgi_client.post(
        "/api/ask_eval",
        json={"question": "hello", "config": {"retrieval_mode": "sparse", "k": 1}},
    )
    assert r.status_code == 503
    assert r.json()["detail"] == "Service unavailable."
    assert "provider unreachable" not in r.text
