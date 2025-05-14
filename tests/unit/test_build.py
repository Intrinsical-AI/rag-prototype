# tests/unit/test_build.py

from pathlib import Path
import csv
import importlib
import pickle
import faiss
from src.settings import settings as st
# NO IMPORTAR SessionLocal o Document directamente aquí si van a cambiar con el reload

import src.db.base as db_base_module
import src.db.models as db_models_module
import scripts.build_index as build_index_module

# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        # Count non-blank lines
        count = 0
        for row in csv.reader(fh):
            if any(field.strip() for field in row): 
                count += 1
        return count


# ------------------------------------------------------------------ #
# TEST 1 – modo sparse (default)
# ------------------------------------------------------------------ #
def test_build_index_sparse(tmp_path, monkeypatch):
    # --- 1) reconfig settings ---
    db_path = tmp_path / "app.db"

    # Don't needed for sparse
    # index_path = tmp_path / "index.faiss"
    # id_map_path = tmp_path / "id_map.pkl"
    # monkeypatch.setattr(st, "index_path", str(index_path))
    # monkeypatch.setattr(st, "id_map_path", str(id_map_path))

    monkeypatch.setattr(st, "sqlite_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(st, "retrieval_mode", "sparse", raising=False)

    # --- 2) recargar engine con nueva URL ---
    importlib.reload(db_base_module)
    importlib.reload(db_models_module)

    # --- 3) ejecutar build_index ---
    build_index_module.main()

    # --- 4) assertions ---
    from src.db.base import SessionLocal
    from src.db.models import Document

    with SessionLocal() as db:
        count_db = db.query(Document).count()

    expected = _count_csv_rows(Path(st.faq_csv)) - 1 # header
    assert count_db == expected, f"Número de filas en DB ({count_db}) debe coincidir con CSV ({expected})"
    assert not (tmp_path / "index.faiss").exists(), "En modo sparse no se genera FAISS" 
    

# ------------------------------------------------------------------ #
# TEST 2 – modo dense
# ------------------------------------------------------------------ #
# ELIMINAR o COMENTAR: @pytest.mark.parametrize("k", [3])
def test_build_index_dense(tmp_path, monkeypatch): # Eliminar ', k'
    # --- 1) reconfig settings ---
    db_path = tmp_path / "app.db"
    index_path = tmp_path / "index.faiss"
    id_map_path = tmp_path / "id_map.pkl"

    monkeypatch.setattr(st, "sqlite_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(st, "index_path", str(index_path))
    monkeypatch.setattr(st, "id_map_path", str(id_map_path))
    monkeypatch.setattr(st, "retrieval_mode", "dense", raising=False)

    # --- 2) recargar engine con nueva URL ---
    importlib.reload(db_base_module)
    importlib.reload(db_models_module)

    # --- 3) ejecutar build_index ---
    build_index_module.main()

    # --- 4) assertions ---
    from src.db.base import SessionLocal
    from src.db.models import Document

    with SessionLocal() as db:
        count_db = db.query(Document).count()
    expected_rows = _count_csv_rows(Path(st.faq_csv)) - 1 # header
    assert count_db == expected_rows, f"Número de filas en DB ({count_db}) debe coincidir con CSV ({expected_rows})"

    assert index_path.exists() and id_map_path.exists(), "FAISS files missing" # Usar variables de path
    faiss_index = faiss.read_index(str(index_path)) # Usar variables de path
    with open(id_map_path, "rb") as fh: # Usar variables de path
        id_map = pickle.load(fh)

    assert faiss_index.ntotal == expected_rows
    assert len(id_map) == expected_rows