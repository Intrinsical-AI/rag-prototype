"""Tests for atomic_write_text / atomic_write_bytes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_rag_backend.infrastructure.persistence.shared.atomic_io import (
    atomic_replace_file,
    atomic_write_bytes,
    atomic_write_text,
)


def test_atomic_write_text_empty_string(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    atomic_write_text(target, "")
    assert target.exists()
    assert target.read_bytes() == b""


def test_atomic_write_text_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    atomic_write_text(target, "hello\nworld")
    assert target.read_text(encoding="utf-8") == "hello\nworld"


def test_atomic_write_text_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.txt"
    atomic_write_text(target, "nested")
    assert target.read_text(encoding="utf-8") == "nested"


def test_atomic_write_bytes_empty(tmp_path: Path) -> None:
    target = tmp_path / "empty.bin"
    atomic_write_bytes(target, b"")
    assert target.exists()
    assert target.read_bytes() == b""


def test_atomic_write_text_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "overwrite.txt"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"


def test_atomic_replace_fsyncs_file_before_rename_and_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_rag_backend.infrastructure.persistence.shared.atomic_io as atomic_io

    source = tmp_path / "source.tmp"
    target = tmp_path / "target.json"
    source.write_bytes(b"durable")
    events: list[str] = []
    original_replace = os.replace

    def record_fsync(descriptor: int) -> None:
        events.append("file_fsync")

    def record_replace(old: Path, new: Path) -> None:
        events.append("replace")
        original_replace(old, new)

    monkeypatch.setattr(atomic_io.os, "fsync", record_fsync)
    monkeypatch.setattr(atomic_io.os, "replace", record_replace)
    monkeypatch.setattr(
        atomic_io, "fsync_directory", lambda directory: events.append("directory_fsync")
    )

    atomic_replace_file(source, target)

    assert events == ["file_fsync", "replace", "directory_fsync"]
    assert target.read_bytes() == b"durable"


def test_new_directory_entries_are_fsynced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import local_rag_backend.infrastructure.persistence.shared.atomic_io as atomic_io

    synced: list[Path] = []
    original_sync = atomic_io.fsync_directory

    def record_sync(path: Path) -> None:
        synced.append(path)
        original_sync(path)

    monkeypatch.setattr(atomic_io, "fsync_directory", record_sync)
    atomic_write_text(tmp_path / "a" / "b" / "receipt.json", "ok")

    assert synced == [tmp_path, tmp_path / "a", tmp_path / "a" / "b"]


def test_atomic_write_cleans_temp_file_if_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_rag_backend.infrastructure.persistence.shared.atomic_io as atomic_io

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("publication failed")

    monkeypatch.setattr(atomic_io, "atomic_replace_file", fail_replace)
    with pytest.raises(OSError, match="publication failed"):
        atomic_write_bytes(tmp_path / "target.json", b"data")
    assert list(tmp_path.glob("target.json.*.tmp")) == []
