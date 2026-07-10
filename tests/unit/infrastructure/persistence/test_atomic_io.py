"""Tests for atomic_write_text / atomic_write_bytes."""

from __future__ import annotations

from pathlib import Path

from local_rag_backend.infrastructure.persistence.shared.atomic_io import (
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
