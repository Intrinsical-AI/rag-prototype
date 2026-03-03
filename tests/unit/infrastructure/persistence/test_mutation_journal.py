from __future__ import annotations

import hashlib
import json

import pytest

from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal


def _record_path_for(root, op_id: str):
    digest = hashlib.sha256(op_id.encode("utf-8")).hexdigest()
    return root / f"{digest}.json"


def test_get_rejects_unknown_state(tmp_path):
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = _record_path_for(journal_dir, "mut:bad-state")
    path.write_text(
        json.dumps(
            {
                "op_id": "mut:bad-state",
                "state": "UNKNOWN_STATE",
                "intent": {},
            }
        ),
        encoding="utf-8",
    )

    journal = FileMutationJournal(journal_dir)
    with pytest.raises(ValueError, match="Invalid mutation journal state"):
        journal.get("mut:bad-state")


def test_list_incomplete_skips_unknown_state_records(tmp_path):
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    (journal_dir / "unknown.json").write_text(
        json.dumps(
            {
                "op_id": "mut:unknown",
                "state": "UNKNOWN_STATE",
                "intent": {},
            }
        ),
        encoding="utf-8",
    )

    journal = FileMutationJournal(journal_dir)
    assert journal.list_incomplete() == []


def test_list_incomplete_skips_blank_op_id_records(tmp_path):
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    (journal_dir / "blank-opid.json").write_text(
        json.dumps(
            {
                "op_id": "",
                "state": "SQL_COMMITTED",
                "intent": {},
            }
        ),
        encoding="utf-8",
    )

    journal = FileMutationJournal(journal_dir)
    assert journal.list_incomplete() == []
