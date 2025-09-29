# tests/unit/settings/test_settings_validators.py
import pytest

from local_rag_backend.settings import Settings


def test_sqlite_url_validator():
    with pytest.raises(ValueError):
        Settings(sqlite_url="postgres://x")


def test_ollama_url_validator():
    with pytest.raises(ValueError):
        Settings(ollama_base_url="localhost:11434")


def test_chunk_overlap_lt_chars_validator():
    """Ensure overlap must be strictly less than chunk size."""
    with pytest.raises(ValueError):
        Settings(ingest_chunk_chars=100, ingest_chunk_overlap=100)
