# ./conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base
from local_rag_backend.infrastructure.persistence.sqlalchemy import sql_
# Import models to ensure they are registered with Base.metadata
from local_rag_backend.infrastructure.persistence.sqlalchemy import models


@pytest.fixture()
def in_memory_sqlite(monkeypatch):
    """
    Crea una BD SQLite en memoria y parchea SessionLocal global
    para que todos los DAOs la usen durante los tests.
    """
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # Crear las tablas que registre nuestro Base (ahora que models está importado)
    db_base.Base.metadata.create_all(bind=engine)

    # Parchar los objetos usados en el código
    monkeypatch.setattr(db_base, "engine", engine)
    monkeypatch.setattr(db_base, "SessionLocal", TestingSessionLocal)
    # También parchear en el módulo sql_ para que SqlDocumentStorage use la sesión de test
    monkeypatch.setattr(sql_, "SessionLocal", TestingSessionLocal)
    
    # Return the session factory for tests that need it explicitly
    return TestingSessionLocal


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
