from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.composition.container import AppContainer
from local_rag_backend.core.ports.contracts import MutationRecord
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal
from local_rag_backend.infrastructure.persistence.sql.base import Base
from local_rag_backend.infrastructure.persistence.sql.document_storage import SqlDocumentStorage
from local_rag_backend.settings import get_settings


def _digests(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


async def test_http_search_reads_current_filtered_store_without_generator_or_history(
    asgi_client, tmp_path, monkeypatch
):
    settings = get_settings()
    engine = create_engine(settings.sqlite_url)
    Base.metadata.create_all(engine)
    repository = SqlDocumentStorage(sessionmaker(bind=engine))
    repository.upsert_documents_by_external_id(
        [
            repository.UpsertDoc(
                external_id="one", content="needle alpha", metadata={"scope": "keep"}
            ),
            repository.UpsertDoc(
                external_id="two", content="needle beta", metadata={"scope": "other"}
            ),
        ]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("search must not initialize or build a generator")

    monkeypatch.setattr(AppContainer, "initialize", forbidden)
    monkeypatch.setattr(AppContainer, "build_generator_from_config", forbidden)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")
    before = _digests(tmp_path)
    response = await asgi_client.post(
        "/api/search",
        json={"question": "needle", "filters": [{"field": "scope", "values": ["keep"]}]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["retrieval_mode"] == "sparse"
    assert [row["document"]["external_id"] for row in response.json()["sources"]] == ["one"]
    assert _digests(tmp_path) == before
    engine.dispose()


async def test_http_search_missing_store_returns_404_without_bootstrapping(
    asgi_client, tmp_path, monkeypatch
):
    database = tmp_path / "absent.sqlite3"
    monkeypatch.setattr(get_settings(), "sqlite_url", f"sqlite:///{database}")
    response = await asgi_client.post("/api/search", json={"question": "needle"})
    assert response.status_code == 404
    assert "Private document store missing" in response.json()["detail"]
    assert not database.exists()


async def test_http_search_refuses_pending_recovery_without_changing_evidence(
    asgi_client, tmp_path, monkeypatch
):
    settings = get_settings()
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")
    engine = create_engine(settings.sqlite_url)
    Base.metadata.create_all(engine)
    journal = FileMutationJournal(settings.get_coordination_dir() / ".mutation_journal")
    journal.upsert(
        MutationRecord(
            op_id="pending",
            state="PREPARED",
            intent={},
            before_image={"docs": [], "existing_tombstones": []},
        )
    )
    before = _digests(tmp_path)
    response = await asgi_client.post("/api/search", json={"question": "needle"})
    assert response.status_code == 503
    assert _digests(tmp_path) == before
    engine.dispose()


async def test_http_mutation_supports_hard_delete_without_tombstoning(asgi_client, monkeypatch):
    monkeypatch.setattr(get_settings(), "retrieval_mode", "sparse")
    upsert = {"upserts": [{"external_id": "hard", "content": "content"}]}
    assert (await asgi_client.post("/api/docs/mutate", json=upsert)).status_code == 200
    response = await asgi_client.post(
        "/api/docs/mutate", json={"op_id": "hard-delete", "hard_delete_external_ids": ["hard"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["deleted_sql"] == 1
    assert response.json()["tombstoned"] == 0
    assert (await asgi_client.post("/api/docs/mutate", json=upsert)).json()["inserted"] == 1
