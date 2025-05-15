# tests/unit/app/test_dependencies.py

import pytest
from unittest.mock import patch, MagicMock, call
import importlib
from pathlib import Path
import csv
import requests
import logging

from src.app import dependencies as app_dependencies_module
from src.settings import settings as global_settings
from src.adapters.generation.openai_chat import OpenAIGenerator
from src.adapters.generation.ollama_chat import OllamaGenerator
from src.adapters.retrieval.sparse_bm25 import SparseBM25Retriever
from src.db.base import SessionLocal as AppSessionLocal
from src.db.models import Document as DbDocument

# ----------------- Fixtures -----------------


@pytest.fixture(autouse=True)
def reset_rag_service_singleton():
    """Ensure the RAG service singleton is reset before and after each test."""
    app_dependencies_module._rag_service = None
    yield
    app_dependencies_module._rag_service = None


@pytest.fixture
def mock_requests_get_fixture():
    """Patch requests.get for Ollama health-check simulation."""
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_path_is_file_fixture():
    """Patch Path.is_file for artifact existence simulation."""
    with patch.object(Path, "is_file") as mock_is_file:
        yield mock_is_file


# ----------------- Tests: _choose_generator logic -----------------


def test_choose_generator_ollama_enabled_and_works(
    monkeypatch, mock_requests_get_fixture
):
    """Ollama enabled and reachable: should use OllamaGenerator."""
    monkeypatch.setattr(global_settings, "ollama_enabled", True)
    monkeypatch.setattr(global_settings, "openai_api_key", None)
    mock_requests_get_fixture.return_value = MagicMock(status_code=200)

    generator = app_dependencies_module._choose_generator()

    assert isinstance(generator, OllamaGenerator)
    mock_requests_get_fixture.assert_called_once_with(
        f"{global_settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2
    )


def test_choose_generator_ollama_enabled_primary_fails_openai_works(
    monkeypatch, mock_requests_get_fixture
):
    """If Ollama check fails and OpenAI key is set, fallback to OpenAIGenerator."""
    monkeypatch.setattr(global_settings, "ollama_enabled", True)
    monkeypatch.setattr(global_settings, "openai_api_key", "sk-testkey")
    mock_requests_get_fixture.side_effect = requests.exceptions.ConnectionError(
        "Ollama down"
    )

    importlib.reload(app_dependencies_module)
    generator = app_dependencies_module._choose_generator()

    assert isinstance(generator, OpenAIGenerator)
    mock_requests_get_fixture.assert_called_once_with(
        f"{global_settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2
    )


def test_choose_generator_ollama_disabled_openai_works(
    monkeypatch, mock_requests_get_fixture
):
    """If Ollama is disabled and OpenAI key is set, use OpenAIGenerator directly."""
    monkeypatch.setattr(global_settings, "ollama_enabled", False)
    monkeypatch.setattr(global_settings, "openai_api_key", "sk-testkey")

    importlib.reload(app_dependencies_module)
    generator = app_dependencies_module._choose_generator()

    assert isinstance(generator, OpenAIGenerator)
    mock_requests_get_fixture.assert_not_called()


def test_choose_generator_ollama_primary_fails_no_openai_key_ollama_fallback_works(
    monkeypatch, mock_requests_get_fixture
):
    """If OpenAI key is not set and Ollama primary fails, should try Ollama fallback."""
    monkeypatch.setattr(global_settings, "ollama_enabled", True)
    monkeypatch.setattr(global_settings, "openai_api_key", None)
    mock_requests_get_fixture.side_effect = [
        requests.exceptions.Timeout("Ollama primary timeout"),
        MagicMock(status_code=200),
    ]

    importlib.reload(app_dependencies_module)
    generator = app_dependencies_module._choose_generator()

    assert isinstance(generator, OllamaGenerator)
    assert mock_requests_get_fixture.call_count == 2
    expected_ollama_tags_url = f"{global_settings.ollama_base_url.rstrip('/')}/api/tags"
    mock_requests_get_fixture.assert_has_calls(
        [
            call(expected_ollama_tags_url, timeout=2),
            call(expected_ollama_tags_url, timeout=2),
        ]
    )


def test_choose_generator_all_fail_raises_runtime_error(
    monkeypatch, mock_requests_get_fixture
):
    """If both Ollama and OpenAI are unavailable, raise RuntimeError."""
    monkeypatch.setattr(global_settings, "ollama_enabled", True)
    monkeypatch.setattr(global_settings, "openai_api_key", None)
    mock_requests_get_fixture.side_effect = requests.exceptions.RequestException(
        "Ollama always fails"
    )

    importlib.reload(app_dependencies_module)
    with pytest.raises(RuntimeError, match="LLM Generator could not be initialized"):
        app_dependencies_module._choose_generator()
    assert mock_requests_get_fixture.call_count == 2


# ----------------- Tests: init_rag_service DB/CSV population and fallback -----------------


def test_init_rag_service_dense_mode_fallback_to_sparse_if_files_missing(
    monkeypatch, mock_path_is_file_fixture, caplog, AppSessionLocal
):
    """Dense mode: if FAISS artifacts missing, fall back to SparseBM25Retriever."""
    with AppSessionLocal() as db:
        db.query(DbDocument).delete()
        db.commit()
        db.add_all([DbDocument(content="doc1"), DbDocument(content="doc2")])
        db.commit()
        assert db.query(DbDocument).count() == 2

    monkeypatch.setattr(global_settings, "retrieval_mode", "dense")
    mock_path_is_file_fixture.return_value = False
    monkeypatch.setattr(global_settings, "openai_api_key", "sk-testkey-dense-fallback")
    monkeypatch.setattr(
        global_settings, "faq_csv", "/tmp/non_existent_faq_for_dense_fallback.csv"
    )

    caplog.set_level(logging.WARNING, logger="src.app.dependencies")
    importlib.reload(app_dependencies_module)
    app_dependencies_module.init_rag_service()
    service = app_dependencies_module.get_rag_service()

    assert service is not None
    assert isinstance(service.retriever, SparseBM25Retriever)
    expected_log_substring = "Falling back to sparse retrieval: dense artifacts missing"
    found_log = any(
        record.name == "src.app.dependencies"
        and record.levelname == "WARNING"
        and expected_log_substring in record.message
        for record in caplog.records
    )
    assert found_log, f"Expected fallback warning log not found. Logs:\n{caplog.text}"


def test_init_rag_service_populates_db_from_csv_if_empty(
    monkeypatch, tmp_path: Path, caplog, AppSessionLocal
):
    """Should populate DB from CSV if empty."""
    with AppSessionLocal() as db:
        db.query(DbDocument).delete()
        db.commit()
        assert db.query(DbDocument).count() == 0

    csv_file_path = tmp_path / "dummy_faq_for_population.csv"
    csv_data = [
        ["question", "answer"],
        ["Q1", "A1: some keywords"],
        ["Q2", "A2: more data"],
    ]
    num_data_rows = len(csv_data) - 1

    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(csv_data)

    monkeypatch.setattr(global_settings, "faq_csv", str(csv_file_path))
    monkeypatch.setattr(global_settings, "csv_has_header", True)
    monkeypatch.setattr(global_settings, "openai_api_key", "sk-testkey-csv-pop")
    monkeypatch.setattr(global_settings, "retrieval_mode", "sparse")

    caplog.set_level(logging.INFO, logger="src.app.dependencies")
    importlib.reload(app_dependencies_module)
    app_dependencies_module.init_rag_service()

    with AppSessionLocal() as db_after_init:
        doc_count = db_after_init.query(DbDocument).count()
        assert doc_count == num_data_rows


def test_init_rag_service_does_not_populate_db_if_not_empty(
    monkeypatch, tmp_path: Path, caplog
):
    """Should not populate DB from CSV if DB already has content."""
    initial_doc_content = "existing document"
    with AppSessionLocal() as db:
        db.query(DbDocument).delete()
        db.add(DbDocument(content=initial_doc_content))
        db.commit()
        assert db.query(DbDocument).count() == 1

    csv_file_path = tmp_path / "dummy_faq_ignored.csv"
    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["QH", "AH"])
        writer.writerow(["NewQ", "NewA"])

    monkeypatch.setattr(global_settings, "faq_csv", str(csv_file_path))
    monkeypatch.setattr(global_settings, "openai_api_key", "sk-testkey")
    monkeypatch.setattr(global_settings, "retrieval_mode", "sparse")

    caplog.set_level(logging.INFO, logger="src.app.dependencies")
    importlib.reload(app_dependencies_module)
    app_dependencies_module.init_rag_service()

    with AppSessionLocal() as db:
        doc_count = db.query(DbDocument).count()
        assert doc_count == 1
        first_doc = db.query(DbDocument).first()
        assert first_doc.content == initial_doc_content

    assert not any(
        record.levelname == "INFO" and "Populating from" in record.message
        for record in caplog.records
        if record.name == "src.app.dependencies"
    ), f"DB population log found, but DB was not empty. Logs:\n{caplog.text}"


def test_get_rag_service_raises_assertion_error_if_not_initialized():
    """Should raise AssertionError if RAG service is accessed before initialization."""
    assert app_dependencies_module._rag_service is None
    with patch.object(
        app_dependencies_module, "init_rag_service"
    ) as mock_init_fallback:
        with pytest.raises(AssertionError, match="RagService has not been initialized"):
            app_dependencies_module.get_rag_service()
        mock_init_fallback.assert_not_called()

    app_dependencies_module._rag_service = None
    with patch.object(app_dependencies_module, "init_rag_service") as mock_init_fails:
        with pytest.raises(AssertionError, match="RagService has not been initialized"):
            app_dependencies_module.get_rag_service()
        mock_init_fails.assert_not_called()


def test_get_rag_service_raises_if_init_fails_to_set_service(monkeypatch):
    """Should raise AssertionError if init_rag_service fails to set singleton."""
    assert app_dependencies_module._rag_service is None

    def mock_init_that_does_not_set_singleton():
        app_dependencies_module._rag_service = None  # Explicitly keep it None

    monkeypatch.setattr(
        app_dependencies_module,
        "init_rag_service",
        mock_init_that_does_not_set_singleton,
    )
    with pytest.raises(AssertionError, match="RagService has not been initialized"):
        app_dependencies_module.get_rag_service()
