from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from local_rag_backend.core.errors import MutationRecoveryRequiredError
from local_rag_backend.core.ports.contracts import MutationRecord
from local_rag_backend.infrastructure.observability.diagnostics import (
    get_incomplete_mutation_records_count,
)
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
    path = _record_path_for(tmp_path / "active", op_id)
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


def _outcome(op_id: str) -> dict[str, object]:
    return {
        "op_id": op_id,
        "inserted": 1,
        "updated": 0,
        "unchanged": 0,
        "deleted_sql": 0,
        "deleted_index": None,
        "tombstoned": 0,
        "missing_external_ids": [],
        "results": [],
    }


def test_committed_receipt_moves_to_done_without_before_image(tmp_path):
    journal = FileMutationJournal(tmp_path)
    op_id = "mut:compact"
    pending = MutationRecord(
        op_id=op_id,
        state="PREPARED",
        intent={},
        before_image={"docs": [], "existing_tombstones": []},
    )
    journal.upsert(pending)
    active = _record_path_for(tmp_path / "active", op_id)
    done = _record_path_for(tmp_path / "done", op_id)
    assert active.is_file()

    journal.upsert(replace(pending, state="COMMITTED", outcome=_outcome(op_id)))

    assert not active.exists()
    assert done.is_file()
    assert journal.get(op_id).before_image is None
    assert json.loads(done.read_text())["before_image"] is None
    assert journal.list_incomplete() == []


@pytest.mark.parametrize("read", ["get", "list_incomplete", "count_incomplete"])
@pytest.mark.parametrize("unversioned", [False, True])
def test_flat_journal_layout_is_refused_without_changing_evidence(tmp_path, read, unversioned):
    op_id = "mut:unsupported"
    journal = FileMutationJournal(tmp_path)
    journal.upsert(
        MutationRecord(op_id=op_id, state="COMMITTED", intent={}, outcome=_outcome(op_id))
    )
    current = _record_path_for(tmp_path / "done", op_id)
    payload = json.loads(current.read_text())
    if unversioned:
        payload.pop("schema_version")
        payload.pop("vector_attempted")
    flat = _record_path_for(tmp_path, op_id)
    flat.write_text(json.dumps(payload))
    current.unlink()
    original = flat.read_bytes()
    reopened = FileMutationJournal(tmp_path)
    with pytest.raises(MutationRecoveryRequiredError, match="Unsupported mutation journal layout"):
        reopened.get(op_id) if read == "get" else getattr(reopened, read)()
    assert flat.read_bytes() == original
    assert not current.exists()


@pytest.mark.parametrize("field", ["schema_version", "vector_attempted"])
def test_unversioned_current_record_is_refused_and_preserved(tmp_path, field):
    op_id = "mut:missing-contract"
    journal = FileMutationJournal(tmp_path)
    journal.upsert(
        MutationRecord(op_id=op_id, state="COMMITTED", intent={}, outcome=_outcome(op_id))
    )
    path = _record_path_for(tmp_path / "done", op_id)
    payload = json.loads(path.read_text())
    payload.pop(field)
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    with pytest.raises(MutationRecoveryRequiredError, match="Unreadable mutation journal"):
        FileMutationJournal(tmp_path).get(op_id)
    assert path.read_bytes() == original


def test_constructing_journal_does_not_change_existing_data(tmp_path):
    root = tmp_path / ".mutation_journal"
    FileMutationJournal(root)
    assert not root.exists()


def test_get_rejects_symlinked_record(tmp_path):
    journal = FileMutationJournal(tmp_path)
    journal.active_dir.mkdir()
    target = tmp_path / "unrelated.txt"
    target.write_text("{}")
    _record_path_for(journal.active_dir, "mut:link").symlink_to(target)
    with pytest.raises(MutationRecoveryRequiredError, match="not a regular file"):
        journal.get("mut:link")


def test_done_receipts_expire_by_age_and_size(tmp_path, monkeypatch):
    import local_rag_backend.infrastructure.persistence.shared.mutation_journal as journal_module

    # Exercise a fresh CI runner whose monotonic clock is below the hourly interval.
    monkeypatch.setattr(journal_module.time, "monotonic", lambda: 120.0)
    journal = FileMutationJournal(tmp_path)
    for index in range(3):
        op_id = f"mut:retention:{index}"
        journal.upsert(
            MutationRecord(op_id=op_id, state="COMMITTED", intent={}, outcome=_outcome(op_id))
        )
    old = _record_path_for(tmp_path / "done", "mut:retention:0")
    os.utime(old, (time.time() - 31 * 86400, time.time() - 31 * 86400))
    journal._last_sweep_monotonic = float("-inf")
    assert journal.list_incomplete() == []
    assert not old.exists()

    kept = _record_path_for(tmp_path / "done", "mut:retention:2")
    monkeypatch.setattr(journal_module, "_DONE_MAX_BYTES", kept.stat().st_size)
    journal._last_sweep_monotonic = float("-inf")
    assert journal.list_incomplete() == []
    assert kept.is_file()
    assert not _record_path_for(tmp_path / "done", "mut:retention:1").exists()


def test_readiness_count_uses_journal_validation(tmp_path):
    root = tmp_path / ".mutation_journal"
    journal = FileMutationJournal(root)
    journal.upsert(
        MutationRecord(
            op_id="mut:readiness",
            state="SQL_COMMITTED",
            intent={},
            before_image={"docs": [], "existing_tombstones": []},
        )
    )
    assert get_incomplete_mutation_records_count(coordination_dir=tmp_path) == 1

    active = _record_path_for(root / "active", "mut:readiness")
    active.write_text("{bad json")
    with pytest.raises(MutationRecoveryRequiredError, match="Unreadable mutation journal"):
        get_incomplete_mutation_records_count(coordination_dir=tmp_path)


def test_readiness_count_rejects_corrupt_completed_receipt(tmp_path):
    root = tmp_path / ".mutation_journal"
    journal = FileMutationJournal(root)
    op_id = "mut:corrupt-done"
    journal.upsert(
        MutationRecord(op_id=op_id, state="COMMITTED", intent={}, outcome=_outcome(op_id))
    )
    assert get_incomplete_mutation_records_count(coordination_dir=tmp_path) == 0
    done = _record_path_for(root / "done", op_id)
    done.write_text("{bad json")
    with pytest.raises(MutationRecoveryRequiredError, match="Unreadable mutation journal"):
        get_incomplete_mutation_records_count(coordination_dir=tmp_path)


def test_readiness_reuses_validation_of_unchanged_receipts(tmp_path, monkeypatch):
    import local_rag_backend.infrastructure.persistence.shared.mutation_journal as journal_module

    root = tmp_path / ".mutation_journal"
    journal = FileMutationJournal(root)
    op_id = "mut:stable-done"
    journal.upsert(
        MutationRecord(op_id=op_id, state="COMMITTED", intent={}, outcome=_outcome(op_id))
    )
    reads = 0
    original_read = journal_module._read_record

    def counted_read(path, **kwargs):
        nonlocal reads
        reads += 1
        return original_read(path, **kwargs)

    monkeypatch.setattr(journal_module, "_read_record", counted_read)
    assert journal.count_incomplete() == 0
    assert reads == 1
    assert journal.count_incomplete() == 0
    assert reads == 1


def test_get_follows_receipt_moved_during_active_read(tmp_path, monkeypatch):
    import local_rag_backend.infrastructure.persistence.shared.mutation_journal as journal_module

    journal = FileMutationJournal(tmp_path)
    op_id = "mut:moved-during-get"
    journal.upsert(
        MutationRecord(
            op_id=op_id,
            state="PREPARED",
            intent={},
            before_image={"docs": [], "existing_tombstones": []},
        )
    )
    active = _record_path_for(journal.active_dir, op_id)
    real_read = journal_module._read_record
    moved = False

    def move_then_read(path, **kwargs):
        nonlocal moved
        if path == active and not moved:
            moved = True
            journal._move_to_done(active, op_id)
        return real_read(path, **kwargs)

    # A terminal payload is written to active before its rename to done/.
    active_payload = json.loads(active.read_text())
    active_payload["state"] = "COMMITTED"
    active_payload["outcome"] = _outcome(op_id)
    active.write_text(json.dumps(active_payload))
    monkeypatch.setattr(journal_module, "_read_record", move_then_read)

    assert journal.get(op_id).state == "COMMITTED"
    assert moved


def test_get_fails_closed_if_active_record_vanishes_without_receipt(tmp_path, monkeypatch):
    import local_rag_backend.infrastructure.persistence.shared.mutation_journal as journal_module

    journal = FileMutationJournal(tmp_path)
    op_id = "mut:lost-active"
    journal.upsert(
        MutationRecord(
            op_id=op_id,
            state="PREPARED",
            intent={},
            before_image={"docs": [], "existing_tombstones": []},
        )
    )
    active = _record_path_for(journal.active_dir, op_id)
    real_read = journal_module._read_record

    def remove_then_read(path, **kwargs):
        if path == active:
            active.unlink()
        return real_read(path, **kwargs)

    monkeypatch.setattr(journal_module, "_read_record", remove_then_read)

    with pytest.raises(MutationRecoveryRequiredError, match="vanished without a receipt"):
        journal.get(op_id)


def test_readiness_accepts_terminal_receipt_moved_during_active_read(tmp_path, monkeypatch):
    import local_rag_backend.infrastructure.persistence.shared.mutation_journal as journal_module

    journal = FileMutationJournal(tmp_path)
    op_id = "mut:moved-during-readiness"
    journal.upsert(
        MutationRecord(op_id=op_id, state="COMMITTED", intent={}, outcome=_outcome(op_id))
    )
    active = _record_path_for(journal.active_dir, op_id)
    done = _record_path_for(journal.done_dir, op_id)
    os.replace(done, active)
    real_read = journal_module._read_record
    moved = False

    def move_then_read(path, **kwargs):
        nonlocal moved
        if path == active and not moved:
            moved = True
            journal._move_to_done(active, op_id)
        return real_read(path, **kwargs)

    monkeypatch.setattr(journal_module, "_read_record", move_then_read)

    assert journal.count_incomplete() == 0
    assert moved


def test_readiness_skips_done_receipt_removed_after_stat(tmp_path, monkeypatch):
    import local_rag_backend.infrastructure.persistence.shared.mutation_journal as journal_module

    journal = FileMutationJournal(tmp_path)
    op_id = "mut:expired-during-readiness"
    journal.upsert(
        MutationRecord(op_id=op_id, state="COMMITTED", intent={}, outcome=_outcome(op_id))
    )
    done = _record_path_for(journal.done_dir, op_id)
    real_read = journal_module._read_record

    def remove_then_read(path, **kwargs):
        if path == done:
            done.unlink()
        return real_read(path, **kwargs)

    monkeypatch.setattr(journal_module, "_read_record", remove_then_read)

    assert journal.count_incomplete() == 0


def test_terminal_move_fsyncs_destination_then_source(tmp_path, monkeypatch):
    import local_rag_backend.infrastructure.persistence.shared.mutation_journal as journal_module

    journal = FileMutationJournal(tmp_path)
    op_id = "mut:fsync-move"
    journal.upsert(
        MutationRecord(
            op_id=op_id,
            state="PREPARED",
            intent={},
            before_image={"docs": [], "existing_tombstones": []},
        )
    )
    active = _record_path_for(journal.active_dir, op_id)
    events: list[tuple[str, Path]] = []
    original_replace = os.replace
    original_sync = journal_module.fsync_directory

    def record_replace(old: Path, new: Path) -> None:
        events.append(("replace", new))
        original_replace(old, new)

    def record_sync(path: Path) -> None:
        events.append(("sync", path))
        original_sync(path)

    monkeypatch.setattr(journal_module.os, "replace", record_replace)
    monkeypatch.setattr(journal_module, "fsync_directory", record_sync)

    journal._move_to_done(active, op_id)

    assert events == [
        ("replace", _record_path_for(journal.done_dir, op_id)),
        ("sync", journal.done_dir),
        ("sync", journal.active_dir),
    ]
