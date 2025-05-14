# tests/conftest.py (o tests/integration/conftest.py si solo aplica a integración)
import sys
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool # MUY IMPORTANTE para SQLite en memoria con FastAPI/multithreading

# --- 1. Add src to PYTHONPATH ---
# Assurs'from src...' working on tests/app when imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.is_dir():
    sys.path.insert(0, str(PROJECT_ROOT)) # Añadir el directorio PADRE de src
else:
    pass 

# --- 2. Parchear Settings Globalmente para la Sesión de Test ANTES de importar la app o dependencias ---
from src.settings import settings

# Memmory BD, default sparse mode, dummy API keys
settings.sqlite_url = "sqlite:///:memory:"      # memory db for all the tests
settings.retrieval_mode = "sparse"              # default for tests
settings.openai_api_key = "sk-dummy-test-key"   # avoid silly errores when OpenAI()
settings.ollama_enabled = False                 # Default para tests

# --- 3. Reconfigurar el Engine y SessionLocal de src.db.base para usar la BD en memoria ---
# Importa db_base AFTER settings.sqlite_url being patched
import src.db.base as db_base_module

# Crear un nuevo engine que apunte a la BD en memoria con StaticPool
# StaticPool es crucial para SQLite en memoria en contextos multi-hilo/async como FastAPI.
# Mantiene una única conexión subyacente por "hilo" (o en este caso, para el pool).
test_engine = create_engine(
    settings.sqlite_url, # Ya es "sqlite:///:memory:"
    connect_args={"check_same_thread": False}, # Necesario para SQLite
    poolclass=StaticPool, # Usar StaticPool
)

# Sobrescribir el engine global en el módulo db_base
db_base_module.engine = test_engine
# Reconfigurar la SessionLocal global para que use el nuevo test_engine
db_base_module.SessionLocal.configure(bind=test_engine)


# --- 4. Crear Esquema de BD y Poblar con Datos de Prueba Conocidos ---
# Importar Base y modelos DESPUÉS de reconfigurar db_base_module.engine
from src.db.models import Base as AppDeclarativeBase # El Base de tus modelos
from src.db.models import Document as DbDocument     # Tu modelo Document

# Crear todas las tablas definidas en AppDeclarativeBase (Document, QaHistory)
AppDeclarativeBase.metadata.create_all(bind=test_engine)

# Poblar la tabla Document con datos conocidos para los tests de integración
# Estos datos deben ser consistentes con lo que los tests esperan recuperar.
# Los IDs aquí son explícitos. Si build_index.py se usara, los IDs serían autoincrementales.
# Para tests de integración donde NO probamos build_index.py, es mejor setear datos explícitos.
with db_base_module.SessionLocal() as db:
    # Limpiar datos de ejecuciones anteriores si la BD no fuera puramente en memoria
    # (StaticPool con :memory: debería ser limpia cada vez, pero por si acaso)
    db.query(DbDocument).delete()
    db.commit()

    # Datos de prueba consistentes con las aserciones en test_api_ask.py
    test_docs_data = [
        {"id": 1, "content": "The refund policy states you can request a refund within 14 days."},
        {"id": 2, "content": "To contact support, please email support@example.com."},
        {"id": 3, "content": "Available features include semantic search and document processing."},
        # Añade más si es necesario para tus tests de retrieval
    ]
    for doc_data in test_docs_data:
        db.add(DbDocument(**doc_data))
    db.commit()

# --- 5. Fixture para Inicializar RagService una vez por sesión con la config de test ---
@pytest.fixture(scope="session", autouse=True)
def initialize_rag_service_for_session():
    """
    Esta fixture se ejecuta una vez por sesión de test, automáticamente.
    Llama a init_rag_service DESPUÉS de que todos los settings y la BD
    hayan sido configurados para el entorno de test.
    """
    # Importar el módulo de dependencias y llamar a init_rag_service
    # Esto asegura que el _rag_service singleton se crea usando los settings parcheados
    # y la BD en memoria ya poblada.
    from src.app import dependencies as app_dependencies
    
    # Es importante que init_rag_service() use el SessionLocal y Document
    # que ahora están vinculados al test_engine y la BD en memoria.
    # El reload de app_dependencies podría ser necesario si este importa settings
    # o db_base a nivel de módulo y queremos que relea los valores parcheados.
    # Sin embargo, como los settings y db_base.engine/SessionLocal se parchean
    # *antes* de esta fixture (a nivel de módulo de conftest), la primera importación
    # de app_dependencies ya debería ver los valores correctos.
    # Si init_rag_service en sí mismo importa settings o db_base directamente, también está bien.
    
    # importlib.reload(app_dependencies) # Opcional, probar sin él primero.
                                       # Podría ser necesario si app_dependencies
                                       # captura settings en el momento de su primera importación.
    
    app_dependencies.init_rag_service() # Esto creará el _rag_service global
    yield
    # No se necesita limpieza aquí si el _rag_service no mantiene recursos abiertos
    # que necesiten cierre explícito al final de la sesión.