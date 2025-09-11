# tests/unit/settings/test_settings_validators.py
import pytest

from local_rag_backend.settings import Settings


def test_sqlite_url_validator():
    with pytest.raises(ValueError):
        Settings(sqlite_url="postgres://x")


def test_ollama_url_validator():
    with pytest.raises(ValueError):
        Settings(ollama_base_url="localhost:11434")
