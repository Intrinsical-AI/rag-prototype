from __future__ import annotations

import subprocess

import pytest
from support import repogpt_fixture


def test_explicit_absent_repogpt_fails(tmp_path):
    absent = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="uv sync --frozen --extra dev"):
        repogpt_fixture.resolve_repogpt_python(absent)


def test_present_repogpt_without_own_environment_fails_actionably(tmp_path, monkeypatch):
    monkeypatch.setattr(repogpt_fixture, "REPOGPT_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="uv sync --frozen --extra dev"):
        repogpt_fixture.load_repogpt_payload(payload_path=tmp_path / "out.json", repo_path=tmp_path)


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
        repogpt_fixture.load_repogpt_payload(payload_path=tmp_path / "out.json", repo_path=tmp_path)
    assert calls[0][0] == str(python)
    assert "--replace-scope" in calls[0]
    assert "--include-tests" in calls[0]


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
        repogpt_fixture.load_repogpt_payload(payload_path=tmp_path / "out.json", repo_path=tmp_path)
    assert caught.value.__cause__ is cause


def test_default_payload_is_versioned_and_needs_no_external_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(repogpt_fixture, "REPOGPT_ROOT", None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Default gate must not launch RepoGPT")

    monkeypatch.setattr(repogpt_fixture.subprocess, "run", forbidden)
    output = tmp_path / "payload.json"
    payload = repogpt_fixture.load_repogpt_payload(payload_path=output)
    assert payload["schema_version"] == "5"
    assert payload["replace_scope"] is True
    assert payload["documents"]
    assert output.read_text() == repogpt_fixture.CANONICAL_FIXTURE.read_text()
