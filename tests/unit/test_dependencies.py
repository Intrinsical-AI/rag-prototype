# tests/unit/app/test_dependencies.py
import pytest
from unittest.mock import patch, MagicMock, call
import importlib
from pathlib import Path
import csv
import requests # Para las excepciones

# Importar el módulo a testear y los settings para monkeypatch
from src.app import dependencies as app_dependencies_module
from src.settings import settings as global_settings # El objeto 'settings' real
from src.adapters.generation.openai_chat import OpenAIGenerator
from src.adapters.generation.ollama_chat import OllamaGenerator
from src.adapters.retrieval.sparse_bm25 import SparseBM25Retriever
from src.adapters.retrieval.dense_faiss import DenseFaissRetriever # Asumiendo que este es el nombre
from src.db.base import SessionLocal as AppSessionLocal # La SessionLocal de la app
from src.db.models import Document as DbDocument # El modelo SQLAlchemy
from src.core.rag import RagService


# --- Fixtures Específicas para este Módulo de Test ---
@pytest.fixture(autouse=True)
def reset_rag_service_singleton():
    """Asegura que el singleton _rag_service se resetea antes de cada test."""
    app_dependencies_module._rag_service = None
    yield
    app_dependencies_module._rag_service = None


@pytest.fixture
def mock_requests_get_fixture():
    with patch("requests.get") as mock_get:
        yield mock_get

@pytest.fixture
def mock_path_is_file_fixture():
    with patch.object(Path, 'is_file') as mock_is_file:
        yield mock_is_file

@pytest.fixture
def mock_logging_fixture():
    # Mockear los loggers específicos usados en dependencies.py
    # El nombre del logger es usualmente el nombre del módulo.
    with patch("src.app.dependencies.logging") as mock_log_module:
        # Puedes devolver un mock_logger específico si necesitas aserciones en él
        # mock_logger_instance = MagicMock()
        # mock_log_module.getLogger.return_value = mock_logger_instance
        yield mock_log_module # o mock_logger_instance


# --- Tests para _choose_generator ---

def test_choose_generator_ollama_enabled_and_works(monkeypatch, mock_requests_get_fixture):
    monkeypatch.setattr(global_settings, "ollama_enabled", True)
    monkeypatch.setattr(global_settings, "openai_api_key", None) # Sin OpenAI
    mock_requests_get_fixture.return_value = MagicMock(status_code=200)

    # Recargar el módulo para que use los settings parcheados si _choose_generator los lee directamente
    # o si importa OllamaGenerator/OpenAIGenerator que leen settings en su __init__.
    importlib.reload(app_dependencies_module)
    generator = app_dependencies_module._choose_generator()

    assert isinstance(generator, OllamaGenerator)
    mock_requests_get_fixture.assert_called_once_with(
        f"{global_settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2
    )

def test_choose_generator_ollama_enabled_primary_fails_openai_works(monkeypatch, mock_requests_get_fixture):
    monkeypatch.setattr(global_settings, "ollama_enabled", True)
    monkeypatch.setattr(global_settings, "openai_api_key", "sk-testkey")
    # Falla la primera llamada a Ollama (health check)
    mock_requests_get_fixture.side_effect = requests.exceptions.ConnectionError("Ollama down")

    importlib.reload(app_dependencies_module)
    generator = app_dependencies_module._choose_generator()

    assert isinstance(generator, OpenAIGenerator)
    # Debería haber intentado Ollama primero
    mock_requests_get_fixture.assert_called_once_with(
        f"{global_settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2
    )

def test_choose_generator_ollama_disabled_openai_works(monkeypatch, mock_requests_get_fixture):
    monkeypatch.setattr(global_settings, "ollama_enabled", False)
    monkeypatch.setattr(global_settings, "openai_api_key", "sk-testkey")

    importlib.reload(app_dependencies_module)
    generator = app_dependencies_module._choose_generator()

    assert isinstance(generator, OpenAIGenerator)
    mock_requests_get_fixture.assert_not_called() # No debería intentar Ollama si está deshabilitado

def test_choose_generator_ollama_primary_fails_no_openai_key_ollama_fallback_works(monkeypatch, mock_requests_get_fixture):
    monkeypatch.setattr(global_settings, "ollama_enabled", True)
    monkeypatch.setattr(global_settings, "openai_api_key", None)
    
    # Primera llamada (primary check) falla, segunda llamada (fallback check) funciona
    mock_requests_get_fixture.side_effect = [
        requests.exceptions.Timeout("Ollama primary timeout"),
        MagicMock(status_code=200)
    ]

    importlib.reload(app_dependencies_module)
    generator = app_dependencies_module._choose_generator()

    assert isinstance(generator, OllamaGenerator)
    assert mock_requests_get_fixture.call_count == 2
    expected_ollama_tags_url = f"{global_settings.ollama_base_url.rstrip('/')}/api/tags"
    mock_requests_get_fixture.assert_has_calls([
        call(expected_ollama_tags_url, timeout=2), # Primary check
        call(expected_ollama_tags_url, timeout=2)  # Fallback check
    ])

def test_choose_generator_all_fail_raises_runtime_error(monkeypatch, mock_requests_get_fixture):
    monkeypatch.setattr(global_settings, "ollama_enabled", True) # Intentará Ollama
    monkeypatch.setattr(global_settings, "openai_api_key", None) # No hay OpenAI
    mock_requests_get_fixture.side_effect = requests.exceptions.RequestException("Ollama always fails")

    importlib.reload(app_dependencies_module)
    with pytest.raises(RuntimeError, match="LLM Generator could not be initialized"):
        app_dependencies_module._choose_generator()
    
    # Debería haber intentado Ollama dos veces (primary y fallback)
    assert mock_requests_get_fixture.call_count == 2


# --- Tests para init_rag_service ---
# Estos tests necesitarán una base de datos en memoria.
# El `conftest.py` de nivel superior debería configurar esto para los tests de integración.
# Si este test_dependencies.py está en `tests/unit/app/`, `conftest.py` global
# (en `tests/`) podría no aplicar su magia de BD en memoria de la misma forma.
# ASUMIMOS que `conftest.py` en `tests/` ya configura una BD en memoria y parchea `settings.sqlite_url`.

def test_init_rag_service_dense_mode_fallback_to_sparse_if_files_missing(
    monkeypatch, mock_path_is_file_fixture, mock_logging_fixture, reset_rag_service_singleton
):
    # Arrange
    # Asegurar que la BD en memoria esté configurada por conftest.py
    # Poblar la BD con algunos datos para que SparseBM25Retriever se pueda inicializar
    with AppSessionLocal() as db: # Usa la SessionLocal de la app, que debería estar parcheada
        db.query(DbDocument).delete() # Limpiar por si acaso
        db.add_all([DbDocument(content="doc1"), DbDocument(content="doc2")])
        db.commit()

    monkeypatch.setattr(global_settings, "retrieval_mode", "dense")
    # No necesitamos setear index_path/id_map_path aquí si mock_path_is_file_fixture siempre devuelve False
    mock_path_is_file_fixture.return_value = False # Simula que los archivos FAISS no existen
    monkeypatch.setattr(global_settings, "openai_api_key", "sk-testkey") # Para que _choose_generator funcione

    # Recargar para que los módulos vean los settings parcheados y para que init_rag_service se ejecute fresco
    importlib.reload(app_dependencies_module)
    
    # Act
    app_dependencies_module.init_rag_service()
    service = app_dependencies_module.get_rag_service()

    # Assert
    assert service is not None
    assert isinstance(service.retriever, SparseBM25Retriever)
    
    # Verificar que se loggeó la advertencia
    # El logger mockeado es 'src.app.dependencies.logging'
    # y la llamada es logging.warning(...)
    # mock_logging_fixture.getLogger().warning.assert_any_call(
    #    match="Dense retrieval mode selected, but FAISS index") -> esto es demasiado específico
    # Mejor verificar si warning fue llamado.
    
    # Construir el mensaje esperado es frágil. Es mejor verificar si se llamó warning
    # y si alguna de las llamadas contiene una subcadena clave.
    found_warning_log = False
    for log_call in mock_logging_fixture.warning.call_args_list:
        args, _ = log_call
        if args and "Falling back to sparse retrieval" in args[0]:
            found_warning_log = True
            break
    assert found_warning_log, "Expected fallback warning log was not found."

    mock_path_is_file_fixture.assert_any_call() # Debería haber chequeado al menos un archivo

def test_init_rag_service_populates_db_from_csv_if_empty(
    monkeypatch, tmp_path: Path, mock_logging_fixture, reset_rag_service_singleton
):
    # Arrange
    # Asegurar que la BD en memoria (parcheada por conftest.py) esté VACÍA
    with AppSessionLocal() as db:
        db.query(DbDocument).delete()
        db.commit()
        assert db.query(DbDocument).count() == 0 # Confirmar que está vacía

    # Crear un dummy_faq.csv
    csv_file_path = tmp_path / "dummy_faq.csv"
    csv_data = [
        ["Question Header", "Answer Header"],
        ["Q1", "A1: some keywords"],
        ["Q2", "A2: more data"],
    ]
    num_data_rows = len(csv_data) -1 # Excluir header

    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    monkeypatch.setattr(global_settings, "faq_csv", str(csv_file_path))
    monkeypatch.setattr(global_settings, "csv_has_header", True)
    monkeypatch.setattr(global_settings, "openai_api_key", "sk-testkey") # Para _choose_generator
    monkeypatch.setattr(global_settings, "retrieval_mode", "sparse") # Para simplificar

    importlib.reload(app_dependencies_module)

    # Act
    app_dependencies_module.init_rag_service()

    # Assert
    # Verificar que la DB fue poblada
    with AppSessionLocal() as db:
        doc_count = db.query(DbDocument).count()
        assert doc_count == num_data_rows
        
        # Opcional: verificar contenido
        docs_content = [d.content for d in db.query(DbDocument).all()]
        assert "q1 a1: some keywords" in docs_content # Asume preprocesamiento (lowercase)
        assert "q2 a2: more data" in docs_content

    # Verificar logs de populación
    found_population_log = False
    for log_call in mock_logging_fixture.info.call_args_list:
        args, _ = log_call
        if args and "Populating from" in args[0]:
            found_population_log = True
            break
    assert found_population_log, "Expected DB population info log was not found."

def test_init_rag_service_does_not_populate_db_if_not_empty(
    monkeypatch, tmp_path: Path, mock_logging_fixture, reset_rag_service_singleton
):
    # Arrange
    # Poblar la BD con un documento para que NO esté vacía
    initial_doc_content = "existing document"
    with AppSessionLocal() as db:
        db.query(DbDocument).delete()
        db.add(DbDocument(content=initial_doc_content))
        db.commit()
        assert db.query(DbDocument).count() == 1

    csv_file_path = tmp_path / "dummy_faq_ignored.csv"
    with open(csv_file_path, "w", newline="", encoding="utf-8") as f: # CSV no debería ser leído
        writer = csv.writer(f)
        writer.writerow(["QH", "AH"])
        writer.writerow(["NewQ", "NewA"])

    monkeypatch.setattr(global_settings, "faq_csv", str(csv_file_path))
    monkeypatch.setattr(global_settings, "openai_api_key", "sk-testkey")
    monkeypatch.setattr(global_settings, "retrieval_mode", "sparse")

    importlib.reload(app_dependencies_module)

    # Act
    app_dependencies_module.init_rag_service() # Debería ver que la DB no está vacía

    # Assert
    # La DB no debería haber cambiado
    with AppSessionLocal() as db:
        doc_count = db.query(DbDocument).count()
        assert doc_count == 1 # Sigue siendo 1
        first_doc = db.query(DbDocument).first()
        assert first_doc.content == initial_doc_content

    # Verificar que NO se loggeó la populación
    found_population_log = False
    for log_call in mock_logging_fixture.info.call_args_list:
        args, _ = log_call
        if args and "Populating from" in args[0]: # No debería encontrar este log
            found_population_log = True
            break
    assert not found_population_log, "DB population log found, but DB was not empty."

def test_get_rag_service_raises_assertion_error_if_not_initialized(reset_rag_service_singleton):
    # Asegurar que _rag_service es None (hecho por la fixture)
    assert app_dependencies_module._rag_service is None
    with pytest.raises(AssertionError, match="RagService not initialised"):
        app_dependencies_module.get_rag_service()