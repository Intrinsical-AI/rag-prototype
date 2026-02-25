# tests/integration/test_build_index_sparse.py
import csv
import importlib

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage
from local_rag_backend.infrastructure.persistence.sql.base import Base
from local_rag_backend.settings import settings


def test_build_index_sparse(tmp_path, monkeypatch):
    f = tmp_path / "faq.csv"
    with f.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["Q", "A"])
        w.writerow(["T", "C"])
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "faq_csv", str(f), raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path}/app.db", raising=False)

    # build_index should run without error and populate the DB.
    from local_rag_backend.scripts import build_index

    importlib.reload(build_index)
    build_index.main()

    # Check database was populated
    eng = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=eng, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=eng)
    docs = SqlDocumentStorage(session_factory=Session).get_all_documents()
    assert len(docs) == 1
    assert "T" in docs[0].content
    assert "c" in docs[0].content.lower()
