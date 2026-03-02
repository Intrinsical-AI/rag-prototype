from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage, base as db_base
from local_rag_backend.infrastructure.persistence.sql.models import Document as DbDocument


def test_ensure_sqlite_schema_compatible_runs_bootstrap_steps(monkeypatch):
    calls: list[tuple[object, ...]] = []
    dummy_engine = object()

    def _create_all(*, bind):
        calls.append(("create_all", bind))

    monkeypatch.setattr(db_base.Base.metadata, "create_all", _create_all, raising=False)

    db_base.ensure_sqlite_schema_compatible(
        engine_to_use=dummy_engine,
        id_map_path="id_map.json",
    )
    assert calls == [
        ("create_all", dummy_engine),
    ]


def test_fresh_schema_contains_doc_id_and_identity_columns(tmp_path):
    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    db_base.ensure_sqlite_schema_compatible(engine_to_use=engine)

    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(documents)")).fetchall()}
        assert {
            "doc_id",
            "external_id",
            "source_id",
            "metadata",
            "content_sha256",
            "chunk_dedup_sha256",
            "created_at",
            "updated_at",
            "content",
        }.issubset(cols)


def test_external_id_unique_constraint(tmp_path):
    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    db_base.ensure_sqlite_schema_compatible(engine_to_use=engine)

    with SessionLocal() as s:
        s.add(DbDocument(doc_id="doc:1", content="a", external_id="X"))
        s.commit()

        s.add(DbDocument(doc_id="doc:2", content="b", external_id="X"))
        try:
            s.commit()
        except IntegrityError:
            s.rollback()
        else:
            raise AssertionError("expected unique external_id constraint")


def test_store_documents_returns_doc_ids(in_memory_sqlite):
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    ids = list(repo.store_documents(["a", "b"]))
    assert len(ids) == 2
    assert all(isinstance(x, str) and x.startswith("doc:") for x in ids)
