from __future__ import annotations

import subprocess

import pytest
from support import repogpt_fixture


def test_only_absent_unconfigured_repogpt_is_optional(tmp_path):
    absent = tmp_path / "missing"
    assert repogpt_fixture.resolve_repogpt_python(absent, configured=False) is None
    with pytest.raises(RuntimeError, match="uv sync --frozen --extra dev"):
        repogpt_fixture.resolve_repogpt_python(absent, configured=True)


def test_present_repogpt_without_own_environment_fails_actionably(tmp_path, monkeypatch):
    monkeypatch.setattr(repogpt_fixture, "REPOGPT_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="uv sync --frozen --extra dev"):
        repogpt_fixture.emit_repogpt_code_units(
            payload_path=tmp_path / "out.json", repo_path=tmp_path
        )


def test_repogpt_uses_its_own_python_and_reports_broken_installation(tmp_path, monkeypatch):
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("placeholder", encoding="utf-8")
    python.chmod(0o700)
    monkeypatch.setattr(repogpt_fixture, "REPOGPT_ROOT", tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="missing dependency")

    monkeypatch.setattr(repogpt_fixture.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="uv sync --frozen --extra dev"):
        repogpt_fixture.emit_repogpt_code_units(
            payload_path=tmp_path / "out.json", repo_path=tmp_path
        )
    assert calls[0][0] == str(python)


def test_repogpt_launch_oserror_retains_preparation_and_cause(tmp_path, monkeypatch):
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("placeholder", encoding="utf-8")
    python.chmod(0o700)
    monkeypatch.setattr(repogpt_fixture, "REPOGPT_ROOT", tmp_path)
    cause = OSError("Exec format error")

    def fail_to_launch(*args, **kwargs):
        raise cause

    monkeypatch.setattr(repogpt_fixture.subprocess, "run", fail_to_launch)
    with pytest.raises(RuntimeError, match="uv sync --frozen --extra dev") as caught:
        repogpt_fixture.emit_repogpt_code_units(
            payload_path=tmp_path / "out.json", repo_path=tmp_path
        )
    assert caught.value.__cause__ is cause
