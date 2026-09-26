from __future__ import annotations

import hashlib
import json

import pytest

from local_rag_backend.core.errors import MutationRecoveryRequiredError
from local_rag_backend.core.ports.contracts import MutationRecord
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal


def _record_path_for(root, op_id: str):
    digest = hashlib.sha256(op_id.encode("utf-8")).hexdigest()
    return root / f"{digest}.json"


@pytest.mark.parametrize("read", ["get", "list_incomplete"])
@pytest.mark.parametrize("damage", ["json", "version", "state", "identity", "vector_attempted"])
def test_corrupt_journal_fails_closed_and_preserves_evidence(tmp_path, read, damage):
    journal = FileMutationJournal(tmp_path)
    op_id = "mut:preserve-evidence"
    journal.upsert(
        MutationRecord(
            op_id=op_id,
            state="SQL_COMMITTED",
            intent={},
            before_image={"docs": [], "existing_tombstones": []},
        )
    )
    path = _record_path_for(tmp_path, op_id)
    payload = json.loads(path.read_text())
    if damage == "json":
        content = "{unreadable"
    else:
        key, value = {
            "version": ("schema_version", 1),
            "state": ("state", "UNKNOWN"),
            "identity": ("op_id", "mut:wrong-name"),
            "vector_attempted": ("vector_attempted", "false"),
        }[damage]
        payload[key] = value
        content = json.dumps(payload)
    path.write_text(content)
    with pytest.raises(MutationRecoveryRequiredError) as error:
        journal.get(op_id) if read == "get" else journal.list_incomplete()
    assert str(path) in str(error.value)
    if damage != "json":
        assert str(payload["op_id"]) in str(error.value)
        assert str(payload["state"]) in str(error.value)
    assert "rebuild cannot repair the journal" in str(error.value)
    assert path.read_text() == content


def test_vector_attempt_marker_survives_journal_roundtrip(tmp_path):
    journal = FileMutationJournal(tmp_path)
    journal.upsert(
        MutationRecord(
            op_id="mut:vector-attempted",
            state="COMPENSATING",
            intent={},
            vector_attempted=True,
            before_image={"docs": [], "existing_tombstones": []},
        )
    )
    assert journal.get("mut:vector-attempted").vector_attempted is True
    assert journal.list_incomplete()[0].vector_attempted is True
