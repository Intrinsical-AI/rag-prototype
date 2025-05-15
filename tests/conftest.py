# tests/conftest.py

import sys
from pathlib import Path
import pytest
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLAlchemySession
from sqlalchemy.pool import StaticPool
import importlib

# 1. Add 'src' to PYTHONPATH if not present
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Patch global app settings for the test session
from src.settings import settings as global_app_settings

# Use a named in-memory SQLite DB (shared cache) for all tests in this session
NAMED_IN_MEMORY_DB_URL = "sqlite:///file:test_rag_db?mode=memory&cache=shared&uri=true"
global_app_settings.sqlite_url = NAMED_IN_MEMORY_DB_URL
global_app_settings.retrieval_mode = "sparse"
global_app_settings.openai_api_key = "sk-dummy-conftest-key"
global_app_settings.ollama_enabled = False

empty_faq_path = PROJECT_ROOT / "tests/data/empty_faq_for_tests.csv"
global_app_settings.faq_csv = str(empty_faq_path)
global_app_settings.csv_has_header = True

empty_faq_path.parent.mkdir(parents=True, exist_ok=True)
if not empty_faq_path.exists():
    import csv
    with open(empty_faq_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["question", "answer"])  # Only header

# 3. Patch DB engine and SessionLocal for src.db.base
import src.db.base as db_base_module
import src.db.models as db_models_module
from src.app import dependencies as app_dependencies_module

_test_engine = create_engine(
    global_app_settings.sqlite_url,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,
)
db_base_module.engine = _test_engine
db_base_module.SessionLocal.configure(bind=_test_engine)

# 4. Session-scoped fixture to create all DB tables once
@pytest.fixture(scope="session", autouse=True)
def create_db_tables_session_scoped():
    """
    Create all DB tables once per session using the test engine.
    """
    db_models_module.Base.metadata.create_all(bind=_test_engine)
    yield
    db_models_module.Base.metadata.drop_all(bind=_test_engine)
    _test_engine.dispose()

# 5. Function-scoped fixture: yields a clean session and wipes tables
@pytest.fixture(scope="function")
def db_session() -> SQLAlchemySession:
    """
    Provides a SQLAlchemy session with all tables emptied before each test.
    """
    session = db_base_module.SessionLocal()
    for table in reversed(db_models_module.Base.metadata.sorted_tables):
        session.execute(table.delete())
    session.commit()
    try:
        yield session
    finally:
        session.close()

# 6. Function-scoped fixture: resets the RAG service singleton before each test
@pytest.fixture(scope="function", autouse=True)
def reset_rag_service_singleton():
    """
    Ensures the _rag_service singleton is None before each test.
    """
    app_dependencies_module._rag_service = None
    yield
    # Optionally re-clear again after test if paranoid

# 7. SessionLocal fixture for direct usage (if needed)
@pytest.fixture(scope="function")
def AppSessionLocal() -> sessionmaker:
    return db_base_module.SessionLocal

# 8. Fixture: Populates DB with test data (for integration tests)
@pytest.fixture(scope="function")
def populated_db_session(db_session: SQLAlchemySession):
    """
    Populates the DB with a standard set of documents for integration tests.
    """
    from src.db.models import Document as DbDocument
    test_docs_data = [
        {"id": 101, "content": "The refund policy states you can request a refund within 14 days."},
        {"id": 102, "content": "To contact support, please email support@example.com."},
        {"id": 103, "content": "Available features include semantic search and document processing."},
    ]
    for doc_data in test_docs_data:
        if not db_session.get(DbDocument, doc_data["id"]):
            db_session.add(DbDocument(**doc_data))
    db_session.commit()
    return db_session

# 9. Fixture: Initializes the RAG service (after DB is populated)
@pytest.fixture(scope="function")
def initialized_rag_service(populated_db_session, monkeypatch):
    """
    Initializes the RAG service singleton for integration tests.
    """
    # Ensure the FAQ CSV exists and points to a valid file (not used since DB is already populated)
    dummy_faq_csv = PROJECT_ROOT / "tests/data/dummy_faq_for_init.csv"
    dummy_faq_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dummy_faq_csv, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["question", "answer"])
        writer.writerow(["Q_init1", "A_init1"])

    monkeypatch.setattr(global_app_settings, "faq_csv", str(dummy_faq_csv))
    app_dependencies_module.init_rag_service()
    service = app_dependencies_module.get_rag_service()
    yield service
    app_dependencies_module._rag_service = None

# 10. Aliases for integration tests if needed
@pytest.fixture(scope="function")
def initialized_rag_service_for_integration(initialized_rag_service):
    return initialized_rag_service

@pytest.fixture(scope="function")
def populated_db_for_integration(populated_db_session):
    return populated_db_session
