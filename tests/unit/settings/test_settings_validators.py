# tests/unit/settings/test_settings_validators.py
from pathlib import Path

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


def test_log_level_is_normalized_to_uppercase():
    s = Settings(log_level="debug")
    assert s.log_level == "DEBUG"


def test_data_dir_is_not_created_as_a_side_effect(tmp_path):
    d = tmp_path / "new-data-dir"
    assert not d.exists()
    s = Settings(data_dir=d)
    assert s.data_dir == d
    assert not d.exists()


def test_coordination_dir_prefers_explicit_data_dir(tmp_path):
    data_dir = tmp_path / "coord-dir"
    s = Settings(data_dir=data_dir, sqlite_url=f"sqlite:///{tmp_path / 'app.db'}")
    assert s.get_coordination_dir() == data_dir.resolve()


def test_coordination_dir_uses_absolute_sqlite_parent_when_data_dir_is_default(tmp_path):
    s = Settings(data_dir=Path("data"), sqlite_url=f"sqlite:///{tmp_path / 'app.db'}")
    assert s.get_coordination_dir() == tmp_path.resolve()
