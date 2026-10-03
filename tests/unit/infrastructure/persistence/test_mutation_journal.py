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


@pytest.mark.parametrize("unversioned", [False, True])
def test_legacy_flat_replay_and_lookup_precedence(tmp_path, unversioned):
    op_id = "mut:legacy"
    journal = FileMutationJournal(tmp_path)
    committed = MutationRecord(
        op_id=op_id,
        state="COMMITTED",
        intent={},
        outcome=_outcome(op_id),
    )
    journal.upsert(committed)
    done = _record_path_for(tmp_path / "done", op_id)
    flat = _record_path_for(tmp_path, op_id)
    flat_payload = json.loads(done.read_text())
    if unversioned:
        flat_payload.pop("schema_version")
        flat_payload.pop("vector_attempted")
    flat.write_text(json.dumps(flat_payload))
    done.unlink()
    old_bytes = flat.read_bytes()

    reopened = FileMutationJournal(tmp_path)
    assert reopened.get(op_id).state == "COMMITTED"
    assert reopened.list_incomplete() == []
    reopened.upsert(replace(committed, intent={"source": "done"}))
    assert reopened.get(op_id).intent == {"source": "done"}
    assert flat.read_bytes() == old_bytes

    reopened.upsert(
        MutationRecord(
            op_id=op_id,
            state="PREPARED",
            intent={},
            before_image={"docs": [], "existing_tombstones": []},
        )
    )
    assert reopened.get(op_id).state == "PREPARED"


def test_incomplete_legacy_flat_record_is_recoverable_without_rewriting_it(tmp_path):
    seed = FileMutationJournal(tmp_path)
    op_id = "mut:old-pending"
    seed.upsert(
        MutationRecord(
            op_id=op_id,
            state="SQL_COMMITTED",
            intent={},
            before_image={"docs": [], "existing_tombstones": []},
        )
    )
    active = _record_path_for(tmp_path / "active", op_id)
    flat = _record_path_for(tmp_path, op_id)
    flat.write_bytes(active.read_bytes())
    active.unlink()
    old_bytes = flat.read_bytes()

    reopened = FileMutationJournal(tmp_path)
    assert [record.op_id for record in reopened.list_incomplete()] == [op_id]
    assert reopened.count_incomplete() == 1

    reopened.upsert(replace(reopened.get(op_id), state="ROLLED_BACK"))
    assert reopened.get(op_id).state == "ROLLED_BACK"
    assert reopened.list_incomplete() == []
    assert reopened.count_incomplete() == 0
    assert flat.read_bytes() == old_bytes

    after_restart = FileMutationJournal(tmp_path)
    assert after_restart.get(op_id).state == "ROLLED_BACK"
    assert after_restart.list_incomplete() == []
    assert after_restart.count_incomplete() == 0
    assert flat.read_bytes() == old_bytes


def test_legacy_terminal_overlay_is_retained_when_other_receipts_expire(tmp_path, monkeypatch):
    import local_rag_backend.infrastructure.persistence.shared.mutation_journal as journal_module

    op_id = "mut:old-pending"
    seed = FileMutationJournal(tmp_path)
    seed.upsert(
        MutationRecord(
            op_id=op_id,
            state="SQL_COMMITTED",
            intent={},
            before_image={"docs": [], "existing_tombstones": []},
        )
    )
    active = _record_path_for(tmp_path / "active", op_id)
    flat = _record_path_for(tmp_path, op_id)
    flat.write_bytes(active.read_bytes())
    active.unlink()

    journal = FileMutationJournal(tmp_path)
    journal.upsert(replace(journal.get(op_id), state="ROLLED_BACK"))
    overlay = _record_path_for(tmp_path / "done", op_id)
    another_op_id = "mut:ordinary-receipt"
    journal.upsert(
        MutationRecord(
            op_id=another_op_id,
            state="COMMITTED",
            intent={},
            outcome=_outcome(another_op_id),
        )
    )
    ordinary = _record_path_for(tmp_path / "done", another_op_id)
    old_timestamp = time.time() - 31 * 86400
    os.utime(overlay, (old_timestamp, old_timestamp))
    os.utime(ordinary, (old_timestamp, old_timestamp))
    monkeypatch.setattr(journal_module, "_DONE_MAX_BYTES", 0)

    assert journal.list_incomplete() == []
    assert overlay.is_file()
    assert not ordinary.exists()
    assert FileMutationJournal(tmp_path).list_incomplete() == []
    assert flat.is_file()


def test_malformed_legacy_flat_record_still_fails_closed(tmp_path):
    flat = _record_path_for(tmp_path, "mut:malformed")
    flat.write_text("{not-json")
    journal = FileMutationJournal(tmp_path)
    with pytest.raises(MutationRecoveryRequiredError, match="Unreadable mutation journal"):
        journal.list_incomplete()
    assert flat.read_text() == "{not-json"


def test_constructing_journal_does_not_change_existing_data(tmp_path):
    root = tmp_path / ".mutation_journal"
    FileMutationJournal(root)
    assert not root.exists()


def test_get_rejects_symlinked_record(tmp_path):
    journal = FileMutationJournal(tmp_path)
    journal.active_dir.mkdir()
    target = tmp_path / "unrelated.json"
    target.write_text("{}")
    _record_path_for(journal.active_dir, "mut:link").symlink_to(target)
    with pytest.raises(MutationRecoveryRequiredError, match="not a regular file"):
        journal.get("mut:link")


def test_terminal_legacy_records_are_parsed_only_once(tmp_path, monkeypatch):
    seed = FileMutationJournal(tmp_path)
    op_id = "mut:seed"
    seed.upsert(MutationRecord(op_id=op_id, state="COMMITTED", intent={}, outcome=_outcome(op_id)))
    payload = _record_path_for(tmp_path / "done", op_id).read_text()
    _record_path_for(tmp_path / "done", op_id).unlink()
    for index in range(1000):
        old_op_id = f"mut:old:{index}"
        old_payload = json.loads(payload)
        old_payload["op_id"] = old_op_id
        old_payload["outcome"]["op_id"] = old_op_id
        _record_path_for(tmp_path, old_op_id).write_text(json.dumps(old_payload))

    import local_rag_backend.infrastructure.persistence.shared.mutation_journal as journal_module

    reads = 0
    original_read = journal_module._read_record

    def counted_read(path, **kwargs):
        nonlocal reads
        reads += 1
        return original_read(path, **kwargs)

    monkeypatch.setattr(journal_module, "_read_record", counted_read)
    journal = FileMutationJournal(tmp_path)
    assert journal.list_incomplete() == []
    assert reads == 1000
    assert journal.list_incomplete() == []
    assert reads == 1000


def test_done_receipts_expire_by_age_and_size(tmp_path, monkeypatch):
    import local_rag_backend.infrastructure.persistence.shared.mutation_journal as journal_module

    journal = FileMutationJournal(tmp_path)
    for index in range(3):
        op_id = f"mut:retention:{index}"
        journal.upsert(
            MutationRecord(op_id=op_id, state="COMMITTED", intent={}, outcome=_outcome(op_id))
        )
    old = _record_path_for(tmp_path / "done", "mut:retention:0")
    os.utime(old, (time.time() - 31 * 86400, time.time() - 31 * 86400))
    journal._last_sweep_monotonic = 0.0
    assert journal.list_incomplete() == []
    assert not old.exists()

    kept = _record_path_for(tmp_path / "done", "mut:retention:2")
    monkeypatch.setattr(journal_module, "_DONE_MAX_BYTES", kept.stat().st_size)
    journal._last_sweep_monotonic = 0.0
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
