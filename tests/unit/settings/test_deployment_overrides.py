from pathlib import Path

import pytest

from local_rag_backend.http.security import enforce_safe_bind_config
from local_rag_backend.settings import load_settings_from_yaml


def test_docker_host_override_reaches_startup_guard(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("app_host: 127.0.0.1\napi_key: null\n")
    monkeypatch.setenv("APP_HOST", "0.0.0.0")
    monkeypatch.delenv("API_KEY", raising=False)
    settings = load_settings_from_yaml(config)
    assert settings.app_host == "0.0.0.0"
    with pytest.raises(RuntimeError, match="Refusing to start"):
        enforce_safe_bind_config(settings)


def test_deployment_overrides_win_over_yaml(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "app_host: 127.0.0.1\nollama_enabled: false\nollama_base_url: http://localhost:11434\n"
    )
    monkeypatch.setenv("APP_HOST", "0.0.0.0")
    monkeypatch.setenv("APP_PORT", "8123")
    monkeypatch.setenv("API_KEY", "fixture-only")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    settings = load_settings_from_yaml(config)
    enforce_safe_bind_config(settings)
    assert settings.app_port == 8123
    assert settings.api_key == "fixture-only"
    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.ollama_enabled is True
