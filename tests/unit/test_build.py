# tests/unit/test_build.py
from pathlib import Path
import csv
import importlib
import pickle

import faiss
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.settings import settings as st
import src.db.base as db_base
from src.db.models import Document
import scripts.build_index as build_index


# ---------- helper ----------
def _count_documents(url: str) -> int:
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        return s.query(Document).count()


# ---------- tests ----------
def test_build_index_sparse(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "app_sparse.db"
    csv_file = tmp_path / "kb_sparse.csv"

    rows = [["question", "answer"], ["q1", "a1"], ["q2", "a2"]]
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, delimiter=";").writerows(rows)

    # patch settings
    monkeypatch.setattr(st, "faq_csv", str(csv_file))
    monkeypatch.setattr(st, "csv_has_header", True)
    monkeypatch.setattr(st, "sqlite_url", f"sqlite:///{db_file}")
    monkeypatch.setattr(st, "retrieval_mode", "sparse")

    importlib.reload(db_base)  # refrezca engine de src.db.base

    build_index.main()

    assert _count_documents(f"sqlite:///{db_file}") == 2  # excluye cabecera


def test_build_index_dense(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "app_dense.db"
    csv_file = tmp_path / "kb_dense.csv"
    index_file = tmp_path / "index.faiss"
    id_map_file = tmp_path / "id_map.pkl"

    rows = [["question", "answer"], ["d1", "a1"], ["d2", "a2"]]
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, delimiter=";").writerows(rows)

    monkeypatch.setattr(st, "faq_csv", str(csv_file))
    monkeypatch.setattr(st, "csv_has_header", True)
    monkeypatch.setattr(st, "sqlite_url", f"sqlite:///{db_file}")
    monkeypatch.setattr(st, "index_path", str(index_file))
    monkeypatch.setattr(st, "id_map_path", str(id_map_file))
    monkeypatch.setattr(st, "retrieval_mode", "dense")

    importlib.reload(db_base)

    build_index.main()

    # DB
    assert _count_documents(f"sqlite:///{db_file}") == 2

    # archivos FAISS
    assert index_file.exists()
    assert id_map_file.exists()

    faiss_index = faiss.read_index(str(index_file))
    with open(id_map_file, "rb") as fh:
        ids = pickle.load(fh)

    assert faiss_index.ntotal == 2
    assert len(ids) == 2
