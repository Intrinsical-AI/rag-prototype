# tests/unit/infrastructure/ingestion/test_discovery_and_factory.py

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.infrastructure.ingestion.loaders.discovery import discover_files
from local_rag_backend.infrastructure.ingestion.loaders.factory import (
    detect_file_format,
    get_loader_for_file,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_discover_files_enforces_limits_and_skips_symlinks(tmp_path: Path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_text("12345", encoding="utf-8")
    (d / "b.txt").write_text("12345", encoding="utf-8")
    (d / "c.txt").write_text("12345", encoding="utf-8")

    # Symlink should be skipped by default.
    (d / "link.txt").symlink_to(d / "a.txt")

    files = list(
        discover_files(
            [d],
            recursive=True,
            follow_symlinks=False,
            max_files=2,
            max_file_bytes=10,
            max_total_bytes=10,
        )
    )
    # Max_total_bytes=10 -> only two 5-byte files fit, but max_files=2 also applies.
    assert len(files) == 2
    assert all(f.name in {"a.txt", "b.txt", "c.txt"} for f in files)
    assert all(not f.is_symlink() for f in files)


def test_discover_files_skips_broken_symlink_and_enforces_max_file_bytes(tmp_path: Path):
    d = tmp_path / "d2"
    d.mkdir()

    big = d / "big.txt"
    big.write_text("x" * 20, encoding="utf-8")

    broken = d / "broken.txt"
    broken.symlink_to(d / "does-not-exist.txt")

    files = list(
        discover_files(
            [d],
            recursive=False,
            follow_symlinks=True,
            max_files=0,
            max_file_bytes=10,
            max_total_bytes=0,
        )
    )
    # big is over max_file_bytes and broken symlink can't be stat'ed -> both skipped.
    assert files == []

    # Direct file input path should be yielded when under limits.
    files2 = list(
        discover_files(
            [big],
            recursive=True,
            follow_symlinks=True,
            max_files=1,
            max_file_bytes=100,
            max_total_bytes=100,
        )
    )
    assert files2 == [big]


def test_discover_files_skips_symlink_directory_input_when_follow_symlinks_is_disabled(
    tmp_path: Path,
):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "inside.txt").write_text("ok", encoding="utf-8")

    symlink_dir = tmp_path / "linked"
    symlink_dir.symlink_to(real_dir, target_is_directory=True)

    files_no_follow = list(
        discover_files(
            [symlink_dir],
            recursive=True,
            follow_symlinks=False,
            max_files=10,
            max_file_bytes=10_000,
            max_total_bytes=10_000,
        )
    )
    assert files_no_follow == []

    files_follow = list(
        discover_files(
            [symlink_dir],
            recursive=True,
            follow_symlinks=True,
            max_files=10,
            max_file_bytes=10_000,
            max_total_bytes=10_000,
        )
    )
    assert len(files_follow) == 1
    assert files_follow[0].name == "inside.txt"


def test_factory_skips_binary_and_detects_markdown_heuristically(tmp_path: Path):
    bin_p = tmp_path / "bin.dat"
    bin_p.write_bytes(b"\x00\x01\x02")
    assert detect_file_format(bin_p, use_magic=False).fmt == "binary"
    assert get_loader_for_file(bin_p, use_magic=False) is None

    md_p = tmp_path / "note.txt"  # misleading extension
    md_p.write_text("# Title\n\n- item\n", encoding="utf-8")
    assert detect_file_format(md_p, use_magic=False).fmt == "markdown"
    loader = get_loader_for_file(md_p, use_magic=False)
    assert loader is not None


def test_detect_file_format_empty_and_extension_hint_paths(tmp_path: Path):
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    det0 = detect_file_format(empty, use_magic=False)
    assert det0.fmt == "unknown"

    # A single-column .csv might not look "tabular" heuristically, but extension should still
    # select the CSV loader (the loader can yield one-column rows).
    onecol = tmp_path / "onecol.csv"
    onecol.write_text("Title\nJust content\n", encoding="utf-8")
    det1 = detect_file_format(onecol, use_magic=False)
    assert det1.fmt == "csv"
    assert get_loader_for_file(onecol, use_magic=False) is not None

    txt = tmp_path / "note.txt"
    txt.write_text("hello world", encoding="utf-8")
    det2 = detect_file_format(txt, use_magic=False)
    assert det2.fmt == "text"
