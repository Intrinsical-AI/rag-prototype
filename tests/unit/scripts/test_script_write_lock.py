import csv
from contextlib import contextmanager
from pathlib import Path

import pytest

from local_rag_backend.scripts.sample_data_ingestion import run_sample_data_ingestion
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
    def _fake_lock(*, coordination_dir=None, timeout_s=None, poll_s=None):
        entered.append(coordination_dir)
        yield

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.concurrency.locks.write_lock.multi_store_write_lock",
        _fake_lock,
        raising=True,
    )

    run_sample_data_ingestion(settings_obj=settings)

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
    def _fake_lock(*, coordination_dir=None, timeout_s=None, poll_s=None):
        entered.append(coordination_dir)
        yield

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.concurrency.locks.write_lock.multi_store_write_lock",
        _fake_lock,
        raising=True,
    )

    run_sample_data_ingestion(
        settings_obj=settings,
        schema_error_message="Unable to ensure SQLite schema before sample-data ingestion.",
    )

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
    def _no_op_lock(*, coordination_dir=None, timeout_s=None, poll_s=None):
        yield

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.concurrency.locks.write_lock.multi_store_write_lock",
        _no_op_lock,
        raising=True,
    )

    def _boom(self, items):
        raise RuntimeError("forced store failure")

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.sql.SqlDocumentStorage.upsert_documents_by_external_id",
        _boom,
        raising=True,
    )

    with pytest.raises(RuntimeError, match="forced store failure"):
        run_sample_data_ingestion(
            settings_obj=settings,
            schema_error_message="Unable to ensure SQLite schema before sample-data ingestion.",
        )


def test_build_index_raises_when_schema_ensure_fails(tmp_path, monkeypatch):
    csv_path = tmp_path / "faq.csv"
    _write_minimal_csv(csv_path)

    monkeypatch.setattr(settings, "faq_csv", str(csv_path), raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path / 'app.db'}", raising=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "coord", raising=False)

    def _boom_schema(*, engine_to_use=None, id_map_path=None):
        raise RuntimeError("forced schema failure")

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.sql.base.ensure_sqlite_schema_compatible",
        _boom_schema,
        raising=True,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Unable to ensure SQLite schema before sample-data ingestion\.",
    ):
        run_sample_data_ingestion(
            settings_obj=settings,
            schema_error_message="Unable to ensure SQLite schema before sample-data ingestion.",
        )
