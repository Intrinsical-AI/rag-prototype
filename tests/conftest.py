# tests/conftest.py
import sys
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLAlchemySession
from sqlalchemy.pool import StaticPool
import importlib
import logging

# --- 1. Add src to PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.is_dir() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic config for test logs

# --- 2. Parchear Settings Globalmente para la Sesión de Test ---
# Import settings first, then patch.
from src.settings import settings as global_app_settings

# Use a named in-memory DB for consistency across the test session
# The 'file:test_rag_db?mode=memory&cache=shared' part is the name.
# 'uri=true' is important for these parameters.
NAMED_IN_MEMORY_DB_URL = "sqlite:///file:test_rag_db?mode=memory&cache=shared&uri=true"

global_app_settings.sqlite_url = NAMED_IN_MEMORY_DB_URL
global_app_settings.retrieval_mode = "sparse"  # Default for most tests
global_app_settings.openai_api_key = "sk-dummy-conftest-key"
global_app_settings.ollama_enabled = False
global_app_settings.faq_csv = str(PROJECT_ROOT / "tests/data/empty_faq_for_tests.csv") # Ensure this file exists or is handled
global_app_settings.csv_has_header = True # Assuming a default

# Create the dummy CSV if it doesn't exist, or ensure your tests handle its absence
empty_faq_path = Path(global_app_settings.faq_csv)
empty_faq_path.parent.mkdir(parents=True, exist_ok=True)
if not empty_faq_path.exists():
    with open(empty_faq_path, 'w', newline='', encoding='utf-8') as f:
        # if csv_has_header is True, write a header
        if global_app_settings.csv_has_header:
            import csv
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["question", "answer"])
    logger.info(f"Created dummy FAQ CSV for tests at {empty_faq_path}")


# --- 3. Reconfigurar el Engine y SessionLocal de src.db.base ---
# Import modules AFTER settings are patched
import src.db.base as db_base_module
import src.db.models as db_models_module
from src.app import dependencies as app_dependencies_module # for resetting singleton

_test_engine = create_engine(
    global_app_settings.sqlite_url,  # This will be the NAMED_IN_MEMORY_DB_URL
    connect_args={"check_same_thread": False},
    poolclass=StaticPool, # StaticPool is crucial for shared in-memory DBs
    echo=False # Set to True for SQL debugging
)
logger.info(f"CONTEST: Created _test_engine (id: {id(_test_engine)}) with URL: {_test_engine.url}")

# Directly assign the test engine and reconfigure SessionLocal
# This is the single source of truth for the DB during tests.
db_base_module.engine = _test_engine
db_base_module.SessionLocal.configure(bind=_test_engine)
logger.info(f"CONTEST: Patched db_base_module.engine (id: {id(db_base_module.engine)})")
logger.info(f"CONTEST: Reconfigured db_base_module.SessionLocal to bind to _test_engine")


# --- 4. Fixture para crear el esquema de BD una vez por sesión ---
@pytest.fixture(scope="session", autouse=True)
def create_db_tables_session_scoped():
    """Crea todas las tablas de la BD una vez por sesión de test, usando el _test_engine."""
    logger.info(f"CONTEST (session scope): Creating DB tables on engine {id(_test_engine)}...")
    db_models_module.Base.metadata.create_all(bind=_test_engine)
    yield
    logger.info(f"CONTEST (session scope): Dropping DB tables on engine {id(_test_engine)}...")
    db_models_module.Base.metadata.drop_all(bind=_test_engine)
    _test_engine.dispose() # Clean up engine resources
    logger.info(f"CONTEST (session scope): Disposed _test_engine.")

# --- 5. Fixture para una sesión de BD limpia (tablas vacías) por CADA TEST ---
@pytest.fixture(scope="function")
def db_session() -> SQLAlchemySession: # Use the actual Session type hint
    """
    Proporciona una sesión de SQLAlchemy y asegura que todas las tablas estén vacías
    antes de que el test se ejecute. La sesión usa el _test_engine configurado.
    """
    session = db_base_module.SessionLocal()
    engine_for_session = session.get_bind()
    logger.info(f"CONTEST (db_session fixture): New session (id: {id(session)}) using engine {id(engine_for_session)} ({engine_for_session.url})")

    # Limpiar tablas
    for table in reversed(db_models_module.Base.metadata.sorted_tables):
        session.execute(table.delete())
    session.commit()
    logger.info(f"CONTEST (db_session fixture): Tables cleared for session {id(session)}.")

    try:
        yield session
    finally:
        session.close()
        logger.info(f"CONTEST (db_session fixture): Closed session {id(session)}.")


# --- 6. Fixture para resetear el singleton de RAG Service ---
@pytest.fixture(scope="function", autouse=True)
def reset_rag_service_singleton():
    """Asegura que el _rag_service es None antes de cada test."""
    # Es importante que esto se ejecute ANTES de tests que podrían llamar a init_rag_service
    if hasattr(app_dependencies_module, '_rag_service'):
        app_dependencies_module._rag_service = None
        logger.info("CONTEST (reset_rag_service_singleton): _rag_service reset to None.")
    yield # El test se ejecuta aquí
    # Opcional: Limpiar después del test si es necesario, pero generalmente
    # el reset al inicio de la siguiente prueba es suficiente.
    # if hasattr(app_dependencies_module, '_rag_service'):
    #     app_dependencies_module._rag_service = None


# --- 7. Fixtures para datos poblados y servicio inicializado (para tests de integración) ---

@pytest.fixture(scope="function")
def AppSessionLocal() -> sessionmaker:
    """Proporciona la SessionLocal configurada para los tests."""
    return db_base_module.SessionLocal


@pytest.fixture(scope="function")
def populated_db_session(db_session: SQLAlchemySession): # Depende de la sesión limpia
    """Puebla la DB con datos de prueba comunes para integración."""
    from src.db.models import Document as DbDocument # Import local

    logger.info(f"CONTEST (populated_db_session): Populating DB for session {id(db_session)}.")
    test_docs_data = [
        {"id": 101, "content": "The refund policy states you can request a refund within 14 days."},
        {"id": 102, "content": "To contact support, please email support@example.com."},
        {"id": 103, "content": "Available features include semantic search and document processing."},
    ]
    for doc_data in test_docs_data:
        # Check if exists to avoid conflicts if called multiple times or if IDs are fixed
        if not db_session.get(DbDocument, doc_data["id"]):
             db_session.add(DbDocument(**doc_data))
    db_session.commit()
    count = db_session.query(DbDocument).count()
    logger.info(f"CONTEST (populated_db_session): DB populated with {count} documents for session {id(db_session)}.")
    return db_session


@pytest.fixture(scope="function")
def initialized_rag_service(populated_db_session, monkeypatch): # Depende de la DB poblada
    """
    Inicializa RagService para tests de integración.
    La DB ya está poblada por populated_db_session.
    El singleton _rag_service ya ha sido reseteado por reset_rag_service_singleton.
    """
    logger.info(f"CONTEST (initialized_rag_service): Initializing RAG service.")

    # Asegurar que el CSV de FAQ para esta inicialización específica es el de prueba (o no existe si así se desea)
    # Esto es importante si init_rag_service intenta poblar desde CSV.
    # Si populated_db_session ya pobló, la condición `db.query(DbDocument).count() == 0` será falsa.
    monkeypatch.setattr(global_app_settings, "faq_csv", str(PROJECT_ROOT / "tests/data/dummy_faq_for_init.csv"))
    dummy_init_faq_path = Path(global_app_settings.faq_csv)
    dummy_init_faq_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dummy_init_faq_path, 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["question", "answer"]) # Header
        writer.writerow(["Q_init1", "A_init1"])
    logger.info(f"CONTEST (initialized_rag_service): Set faq_csv to {dummy_init_faq_path} for this init.")

    app_dependencies_module.init_rag_service() # Esto usará la DB poblada y los settings actuales
    service = app_dependencies_module.get_rag_service()
    logger.info(f"CONTEST (initialized_rag_service): RAG service initialized (id: {id(service)}).")

    yield service

    # Limpieza del singleton después del test (aunque reset_rag_service_singleton lo hará al inicio del siguiente)
    app_dependencies_module._rag_service = None
    logger.info(f"CONTEST (initialized_rag_service): RAG service singleton reset post-test.")

# Alias para la fixture que usa el test de integración original, si es necesario
@pytest.fixture(scope="function")
def initialized_rag_service_for_integration(initialized_rag_service):
    return initialized_rag_service

# Alias para la fixture que usa el test de integración original, si es necesario
@pytest.fixture(scope="function")
def populated_db_for_integration(populated_db_session):
    return populated_db_session