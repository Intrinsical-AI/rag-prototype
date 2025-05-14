# tests/conftest.py
import sys
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import importlib

# --- 1. Add src to PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.is_dir() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 2. Parchear Settings Globalmente para la Sesión de Test ---
from src.settings import settings as global_app_settings # Renombrar

global_app_settings.retrieval_mode = "sparse"
global_app_settings.openai_api_key = "sk-dummy-conftest-key"
global_app_settings.ollama_enabled = False
global_app_settings.sqlite_url = "sqlite:///:memory:"
global_app_settings.faq_csv = "/tmp/non_existent_default_faq_for_tests.csv" # O un path a un CSV vacío de test

# --- 3. Reconfigurar el Engine y SessionLocal de src.db.base ---
import src.db.base as db_base_module
import src.db.models as db_models_module # Importar para Base

# Forzar recarga para asegurar que usan settings parcheados
importlib.reload(db_base_module)
importlib.reload(db_models_module) # Si Base se define aquí y usa engine

_test_engine = create_engine(
    global_app_settings.sqlite_url,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

db_base_module.engine = _test_engine
db_base_module.SessionLocal.configure(bind=_test_engine) # Reconfigura la SessionLocal existente

# --- 4. Fixture para crear el esquema de BD una vez por sesión ---
@pytest.fixture(scope="session", autouse=True)
def create_db_tables():
    """Crea todas las tablas de la BD una vez por sesión de test."""
    # AppDeclarativeBase es el Base de tus modelos SQLAlchemy
    db_models_module.Base.metadata.create_all(bind=_test_engine)
    yield
    # Opcional: db_models_module.Base.metadata.drop_all(bind=_test_engine)

# --- 5. Fixture para una sesión de BD limpia por CADA TEST ---
@pytest.fixture(scope="function")
def db_session() -> sessionmaker:
    session = db_base_module.SessionLocal()
    for table in reversed(db_models_module.Base.metadata.sorted_tables):
        session.execute(table.delete()) # LIMPIA TABLAS
    session.commit()
    try:
        yield session
    finally:
        session.close()

# --- 6. ELIMINA LA SIGUIENTE FIXTURE COMPLETAMENTE ---
# @pytest.fixture(scope="session", autouse=True)
# def initialize_rag_service_for_session():
#    ... ESTO ESTABA CAUSANDO PROBLEMAS ...

# --- 7. (OPCIONAL PERO RECOMENDADO) Fixture para tests de integración que necesitan el servicio listo ---
@pytest.fixture(scope="function")
def populated_db_session(db_session: sessionmaker): # Depende de la sesión limpia
    """Puebla la DB con datos de prueba comunes para integración."""
    from src.db.models import Document as DbDocument # Import local para evitar problemas de import circular

    test_docs_data = [
        {"id": 1, "content": "The refund policy states you can request a refund within 14 days."},
        {"id": 2, "content": "To contact support, please email support@example.com."},
        {"id": 3, "content": "Available features include semantic search and document processing."},
    ]
    for doc_data in test_docs_data:
        db_session.add(DbDocument(**doc_data))
    db_session.commit()
    return db_session # Devuelve la sesión ya poblada

@pytest.fixture(scope="function")
def initialized_rag_service_for_integration(populated_db_session, monkeypatch): # Depende de la DB poblada
    """Inicializa RagService para tests de integración. La DB ya está poblada."""
    from src.app import dependencies as app_dependencies

    # Asegurar que _rag_service esté limpio antes de esta inicialización específica
    app_dependencies._rag_service = None
    
    # Aplicar cualquier monkeypatch a settings específico para la integración aquí si es necesario
    # monkeypatch.setattr(global_app_settings, "retrieval_mode", "sparse") # Ejemplo

    # Recargar el módulo de dependencias para que tome los settings correctos
    # y para asegurar que init_rag_service opera en un estado limpio del singleton _rag_service
    importlib.reload(app_dependencies)
    
    app_dependencies.init_rag_service() # Esto usará la DB poblada por populated_db_session
    service = app_dependencies.get_rag_service()
    
    yield service # El test de integración usa este servicio

    # Limpieza del singleton después del test
    app_dependencies._rag_service = None

@pytest.fixture(scope="function")
def populated_db_for_integration(db_session: sessionmaker): # db_session ya limpió
    """Puebla la DB con datos para tests de integración (IDs 1, 2, 3)."""
    from src.db.models import Document as DbDocument
    test_docs_data = [
        {"id": 1, "content": "The refund policy states you can request a refund within 14 days."},
        {"id": 2, "content": "To contact support, please email support@example.com."},
        {"id": 3, "content": "Available features include semantic search and document processing."},
    ]
    for doc_data in test_docs_data:
        db_session.add(DbDocument(**doc_data))
    db_session.commit()
    # En este punto, la DB tiene 3 documentos.
    # El lifespan de la app ahora llamará a init_rag_service y encontrará estos documentos.
    yield db_session # Opcional, si el test necesita la sesión