from __future__ import annotations

from pathlib import Path

import pytest

from local_rag_backend.composition.factory import reset_app_context
from local_rag_backend.core.ports.contracts import MutationRecord
from local_rag_backend.core.use_cases.docs_mutation_contracts import (
    MutationIntent,
    MutationUpsertInput,
    intent_to_dict,
)
from local_rag_backend.http.main import app
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import (
    FileMutationJournal,
)
from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage
from local_rag_backend.settings import settings


def _journal_for_coordination_dir(coordination_dir: Path) -> FileMutationJournal:
    return FileMutationJournal(coordination_dir / ".mutation_journal")


@pytest.mark.integration
async def test_startup_recovery_rolls_back_sql_committed_record_with_file_journal(
    in_memory_sqlite,
    monkeypatch,
    tmp_path: Path,
) -> None:
    _ = in_memory_sqlite
    isolated_data_dir = tmp_path / "data_ok"
    isolated_data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "data_dir", isolated_data_dir, raising=False)
    monkeypatch.setattr(settings, "mutation_recovery_enabled", True, raising=False)
    monkeypatch.setattr(settings, "mutation_recovery_interval_s", 3600.0, raising=False)
    reset_app_context()

    external_id = "doc:e2e:startup-recovery"
    SqlDocumentStorage().upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id=external_id, content="payload")]
    )
    assert any(d.external_id == external_id for d in SqlDocumentStorage().get_all_documents())

    op_id = "mut:e2e:startup-recovery"
    intent = intent_to_dict(
        MutationIntent(
            op_id=op_id,
            upserts=(MutationUpsertInput(external_id=external_id, content="payload"),),
            source="test:e2e:startup-recovery",
        )
    )
    journal = _journal_for_coordination_dir(settings.get_coordination_dir())
    journal.upsert(
        MutationRecord(
            op_id=op_id,
            state="SQL_COMMITTED",
            intent=intent,
            before_image={"docs": [], "existing_tombstones": []},
        )
    )

    async with app.router.lifespan_context(app):
        pass

    assert all(d.external_id != external_id for d in SqlDocumentStorage().get_all_documents())
    recovered = journal.get(op_id)
    assert recovered is not None
    assert recovered.state == "ROLLED_BACK"


@pytest.mark.integration
async def test_startup_recovery_persists_failed_needs_recovery_when_rollback_fails(
    in_memory_sqlite,
    monkeypatch,
    tmp_path: Path,
) -> None:
    _ = in_memory_sqlite
    isolated_data_dir = tmp_path / "data_fail"
    isolated_data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "data_dir", isolated_data_dir, raising=False)
    monkeypatch.setattr(settings, "mutation_recovery_enabled", True, raising=False)
    monkeypatch.setattr(settings, "mutation_recovery_interval_s", 3600.0, raising=False)

    def _fail_hard_delete(self: SqlDocumentStorage, external_ids: object) -> int:
        raise RuntimeError("forced rollback fail")

    monkeypatch.setattr(
        SqlDocumentStorage,
        "hard_delete_by_external_ids",
        _fail_hard_delete,
        raising=True,
    )
    reset_app_context()

    external_id = "doc:e2e:startup-recovery-fail"
    SqlDocumentStorage().upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id=external_id, content="payload")]
    )

    op_id = "mut:e2e:startup-recovery-fail"
    intent = intent_to_dict(
        MutationIntent(
            op_id=op_id,
            upserts=(MutationUpsertInput(external_id=external_id, content="payload"),),
            source="test:e2e:startup-recovery-fail",
        )
    )
    journal = _journal_for_coordination_dir(settings.get_coordination_dir())
    journal.upsert(
        MutationRecord(
            op_id=op_id,
            state="SQL_COMMITTED",
            intent=intent,
            before_image={"docs": [], "existing_tombstones": []},
        )
    )

    async with app.router.lifespan_context(app):
        pass

    # Rollback failed, so the SQL row remains and journal records failed recovery.
    assert any(d.external_id == external_id for d in SqlDocumentStorage().get_all_documents())
    recovered = journal.get(op_id)
    assert recovered is not None
    assert recovered.state == "FAILED_NEEDS_RECOVERY"
    assert recovered.error is not None
    assert "forced rollback fail" in recovered.error
