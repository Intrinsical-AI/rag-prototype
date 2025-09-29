# tests/unit/app/test_api_router_more.py

import pytest
from fastapi.testclient import TestClient

from local_rag_backend.app import dependencies as deps
from local_rag_backend.app.main import app
from local_rag_backend.settings import settings

client = TestClient(app)


def test_health_endpoint_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_ready_endpoint_ok(monkeypatch):
    class _Dummy:
        pass

    # Ensure at least one provider is available
    monkeypatch.setattr(settings, "openai_api_key", "x", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    app.dependency_overrides[deps.get_rag_service] = lambda: _Dummy()
    try:
        r = client.get("/api/ready")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"
        assert "database" in data["checks"]
        assert "rag_service" in data["checks"]
        assert "llm_providers" in data["checks"]
    finally:
        app.dependency_overrides.pop(deps.get_rag_service, None)


def test_ready_endpoint_not_ready_no_llm(monkeypatch):
    class _Dummy:
        pass

    # No providers configured
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    app.dependency_overrides[deps.get_rag_service] = lambda: _Dummy()
    try:
        r = client.get("/api/ready")
        assert r.status_code == 503
        payload = r.json()
        assert payload["detail"]["status"] == "not_ready"
        assert "llm_providers" in payload["detail"]["checks"]
    finally:
        app.dependency_overrides.pop(deps.get_rag_service, None)


def test_templates_endpoint():
    r = client.get("/api/templates")
    assert r.status_code == 200
    arr = r.json()
    assert isinstance(arr, list) and len(arr) >= 3
    names = {t["name"] for t in arr}
    assert {"default", "ollama"}.issubset(names)


def test_config_endpoint_providers(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "key", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", True, raising=False)
    monkeypatch.setattr(settings, "openrouter_enabled", True, raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "k", raising=False)

    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    prov = set(data.get("available_providers", []))
    assert {"openai", "ollama", "openrouter"}.issubset(prov)
    assert data["retrieval_mode"] in {"sparse", "dense", "hybrid"}


def test_ask_eval_invalid_config(monkeypatch):
    payload = {
        "question": "q",
        "config": {
            "retrieval_mode": "invalid-mode",
            "k": 3,
        },
    }
    r = client.post("/api/ask_eval", json=payload)
    assert r.status_code == 400
    assert "Invalid config" in r.json()["detail"]


@pytest.mark.parametrize(
    "config,expected_status",
    [
        ({"retrieval_mode": "invalid", "k": 3}, 400),                 # invalid mode (runtime validation)
        ({"retrieval_mode": "sparse", "k": 0}, 422),                  # k too small (schema)
        ({"retrieval_mode": "dense", "k": 11}, 422),                  # k too large (schema)
        ({"retrieval_mode": "hybrid", "k": 3, "hybrid_alpha": -0.1}, 422),  # schema
        ({"retrieval_mode": "hybrid", "k": 3, "hybrid_alpha": 1.1}, 422),   # schema
        ({"retrieval_mode": "sparse", "k": 3, "temperature": -0.5}, 400),   # runtime validation
        ({"retrieval_mode": "sparse", "k": 3, "temperature": 2.5}, 400),    # runtime validation
        ({"retrieval_mode": "sparse", "k": 3, "max_tokens": 0}, 400),       # runtime validation (we clamp via validate_rag_config)
        ({"retrieval_mode": "sparse", "k": 3, "max_tokens": 999999}, 400),  # runtime validation
    ],
)
def test_ask_eval_invalid_config_parametrized(config, expected_status):
    payload = {"question": "q", "config": config}
    r = client.post("/api/ask_eval", json=payload)
    assert r.status_code == expected_status
    if expected_status == 400:
        assert "Invalid config" in r.json().get("detail", "")


def test_openrouter_generate_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_enabled", False, raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", None, raising=False)
    r = client.post(
        "/api/openrouter/generate",
        json={
            "model": None,
            "system_instruction": "sys",
            "user_content": "hi",
        },
    )
    assert r.status_code == 400


def test_dependencies_no_llm(monkeypatch):
    deps.get_rag_service.cache_clear()
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    with pytest.raises(RuntimeError):
        deps.get_rag_service()


def test_metrics_endpoint_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_monitoring", False, raising=False)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/plain")
    assert r.text.startswith("# Monitoring disabled")
