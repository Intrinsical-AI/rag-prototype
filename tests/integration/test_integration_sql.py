from contextlib import suppress

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage


@pytest.fixture(scope="function")
def in_memory_db(tmp_path):
    # Crea un engine SQLite en disco temporal (evita conflictos de memoria en multihilo)
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # Crear tablas
    Base.metadata.create_all(bind=engine)

    try:
        yield session_factory
    finally:
        # Close all sessions and dispose the engine to release file handles on Windows
        with suppress(Exception):
            session_factory.close_all()
        with suppress(Exception):
            engine.dispose()
        # teardown: borrar archivo
        db_file.unlink(missing_ok=True)


def test_add_and_get_documents(in_memory_db):
    session_factory = in_memory_db
    # 1) Instanciar el storage con la factoría
    storage = SqlDocumentStorage(session_factory=session_factory)

    # 2) Insertar varios textos
    texts = ["Doc uno", "Doc dos", "Doc tres"]
    ids = storage.store_documents(texts)
    assert len(ids) == 3
    assert all(isinstance(i, int) for i in ids)

    # 3) Recuperar por IDs
    docs = storage.get(ids)
    # Debe preservar orden e ids
    assert [d.id for d in docs] == ids
    assert [d.content for d in docs] == texts

    # 4) get_all también
    all_docs = storage.get_all_documents()
    assert len(all_docs) == 3
    assert {d.content for d in all_docs} == set(texts)
