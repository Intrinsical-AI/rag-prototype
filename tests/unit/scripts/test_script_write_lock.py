import csv
import importlib
from contextlib import contextmanager
from pathlib import Path

import pytest

from local_rag_backend.settings import settings


def _write_minimal_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Q", "A"])
        writer.writerow(["t", "c"])


def test_bootstrap_main_uses_multi_store_write_lock(tmp_path, monkeypatch):
    csv_path = tmp_path / "faq.csv"
    _write_minimal_csv(csv_path)

    monkeypatch.setattr(settings, "faq_csv", str(csv_path), raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path / 'app.db'}", raising=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "coord", raising=False)

    entered: list[Path | None] = []

    @contextmanager
    def _fake_lock(*, coordination_dir=None):
        entered.append(coordination_dir)
        yield

    monkeypatch.setattr(
        "local_rag_backend.core.services.write_lock.multi_store_write_lock",
        _fake_lock,
        raising=True,
    )

    from local_rag_backend.scripts import bootstrap

    importlib.reload(bootstrap)
    bootstrap.main(settings=settings)

    assert entered
    assert entered == [settings.get_coordination_dir()]


def test_build_index_uses_multi_store_write_lock_for_sparse_storage(tmp_path, monkeypatch):
    csv_path = tmp_path / "faq.csv"
    _write_minimal_csv(csv_path)

    monkeypatch.setattr(settings, "faq_csv", str(csv_path), raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path / 'app.db'}", raising=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "coord", raising=False)

    entered: list[Path | None] = []

    @contextmanager
    def _fake_lock(*, coordination_dir=None):
        entered.append(coordination_dir)
        yield

    monkeypatch.setattr(
        "local_rag_backend.core.services.write_lock.multi_store_write_lock",
        _fake_lock,
        raising=True,
    )

    from local_rag_backend.scripts import build_index

    importlib.reload(build_index)
    build_index.main()

    assert entered
    assert entered == [settings.get_coordination_dir()]


def test_build_index_raises_when_storage_fails(tmp_path, monkeypatch):
    csv_path = tmp_path / "faq.csv"
    _write_minimal_csv(csv_path)

    monkeypatch.setattr(settings, "faq_csv", str(csv_path), raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path / 'app.db'}", raising=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "coord", raising=False)

    @contextmanager
    def _no_op_lock(*, coordination_dir=None):
        yield

    monkeypatch.setattr(
        "local_rag_backend.core.services.write_lock.multi_store_write_lock",
        _no_op_lock,
        raising=True,
    )

    def _boom(self, texts):
        raise RuntimeError("forced store failure")

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.sql.alchemy_engine.SqlDocumentStorage.store_documents",
        _boom,
        raising=True,
    )

    from local_rag_backend.scripts import build_index

    importlib.reload(build_index)
    with pytest.raises(RuntimeError, match="forced store failure"):
        build_index.main()


def test_build_index_raises_when_schema_ensure_fails(tmp_path, monkeypatch):
    csv_path = tmp_path / "faq.csv"
    _write_minimal_csv(csv_path)

    monkeypatch.setattr(settings, "faq_csv", str(csv_path), raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path / 'app.db'}", raising=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "coord", raising=False)

    def _boom_schema(*, engine_to_use=None):
        raise RuntimeError("forced schema failure")

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.sql.base.ensure_sqlite_documents_identity_columns",
        _boom_schema,
        raising=True,
    )

    from local_rag_backend.scripts import build_index

    importlib.reload(build_index)
    with pytest.raises(RuntimeError, match=r"Unable to ensure SQLite schema before build-index\."):
        build_index.main()
