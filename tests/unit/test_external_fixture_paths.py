from __future__ import annotations

from pathlib import Path

import pytest
from support.external_paths import configured_directory, configured_file


def test_unset_external_path_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXTERNAL_FIXTURE", raising=False)

    assert configured_file("EXTERNAL_FIXTURE") is None
    assert configured_directory("EXTERNAL_FIXTURE") is None


def test_configured_external_file_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_file = tmp_path / "fixture.jsonl"
    fixture_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("EXTERNAL_FILE", str(fixture_file))
    monkeypatch.setenv("EXTERNAL_DIRECTORY", str(tmp_path))

    assert configured_file("EXTERNAL_FILE") == fixture_file
    assert configured_directory("EXTERNAL_DIRECTORY") == tmp_path


def test_invalid_explicit_external_path_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTERNAL_FIXTURE", "/definitely/missing/external-fixture")

    with pytest.raises(RuntimeError, match="EXTERNAL_FIXTURE must point"):
        configured_file("EXTERNAL_FIXTURE")
