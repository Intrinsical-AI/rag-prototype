from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage, base as db_base
from local_rag_backend.infrastructure.persistence.sql.models import Document as DbDocument


def test_ensure_sqlite_schema_current_runs_bootstrap_steps(monkeypatch):
    calls: list[tuple[object, ...]] = []
    dummy_engine = object()

    def _create_all(*, bind):
        calls.append(("create_all", bind))

    monkeypatch.setattr(db_base.Base.metadata, "create_all", _create_all, raising=False)

    db_base.ensure_sqlite_schema_current(engine_to_use=dummy_engine)
    assert calls == [
        ("create_all", dummy_engine),
    ]


def test_fresh_schema_contains_doc_id_and_identity_columns(tmp_path):
    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    db_base.ensure_sqlite_schema_current(engine_to_use=engine)

    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(documents)")).fetchall()}
        assert {
            "doc_id",
            "external_id",
            "source_id",
            "scope",
            "snapshot_id",
            "metadata",
            "content_sha256",
            "created_at",
            "updated_at",
            "content",
        }.issubset(cols)
        assert "chunk_dedup_sha256" not in cols


def test_external_id_unique_constraint(tmp_path):
    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    db_base.ensure_sqlite_schema_current(engine_to_use=engine)

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


def test_existing_nullable_chunk_dedup_column_does_not_block_upserts(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'legacy.db'}", connect_args={"check_same_thread": False}
    )
    db_base.ensure_sqlite_schema_current(engine_to_use=engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE documents ADD COLUMN chunk_dedup_sha256 TEXT"))

    repo = SqlDocumentStorage(
        session_factory=sessionmaker(bind=engine, autocommit=False, autoflush=False)
    )
    results, _, _ = repo.upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id="existing", content="  Raw text  ")]
    )
    assert repo.get([results[0].id])[0].content == "  Raw text  "
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE documents SET chunk_dedup_sha256 = 'legacy' WHERE external_id = 'existing'"
            )
        )

    by_id = repo.snapshot_by_ids([results[0].id])
    by_external_id = repo.snapshot_by_external_ids(["existing"])
    assert by_id == by_external_id
    assert by_id[0]["content"] == "  Raw text  "
    assert "chunk_dedup_sha256" not in by_id[0]


def test_store_documents_returns_doc_ids(in_memory_sqlite):
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    ids = list(repo.store_documents(["a", "b"]))
    assert len(ids) == 2
    assert all(isinstance(x, str) and x.startswith("doc:") for x in ids)
