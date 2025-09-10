# ./conftest.py
import pytest
from contextlib import suppress
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base

# Import models to ensure they are registered with Base.metadata
from local_rag_backend.infrastructure.persistence.sqlalchemy import sql_


@pytest.fixture()
def in_memory_sqlite(monkeypatch):
    """
    Crea una BD SQLite en memoria y parchea SessionLocal global
    para que todos los DAOs la usen durante los tests.
    """
    # Use StaticPool so all sessions share the same in-memory database connection
    # and add check_same_thread=False for sqlite thread-safety in tests
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # Crear las tablas que registre nuestro Base (ahora que models está importado)
    db_base.Base.metadata.create_all(bind=engine)

    # Parchar los objetos usados en el código
    monkeypatch.setattr(db_base, "engine", engine)
    monkeypatch.setattr(db_base, "SessionLocal", TestingSessionLocal)
    # También parchear en el módulo sql_ para que SqlDocumentStorage use la sesión de test
    monkeypatch.setattr(sql_, "SessionLocal", TestingSessionLocal)

    # Yield the session factory for tests that need it explicitly
    try:
        yield TestingSessionLocal
    finally:
        # Ensure all sessions are closed
        with suppress(Exception):
            TestingSessionLocal.close_all()
        # Dispose engine to close underlying connection and avoid ResourceWarning
        engine.dispose()


class DummyFaissIndex:
    def __init__(self, index_path, id_map_path, dim=None):  # <--- dim opcional
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.dim = 4  # igual a DummyEmbedder
        self.id_map = []

    def add_to_index(self, ids, vecs):
        self.id_map.extend(ids)

    def search(self, q, k):
        return ([0], [0.0])

    def save(self):
        pass
