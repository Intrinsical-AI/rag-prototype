"""Exercise actual Make security contracts with controlled tools and no network."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("RAG_SECURITY_SUBJECT_ROOT", str(DEFAULT_ROOT)))
HELPER = Path(
    os.environ.get("RAG_SECURITY_OUTCOME_HELPER", str(ROOT / "scripts/security_result.py"))
)

FAKE_UV = r"""
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["SECURITY_TEST_LOG"]).open("a") as stream:
    stream.write(json.dumps(args) + "\n")
scenario = os.environ["SECURITY_TEST_SCENARIO"]
root = Path(os.environ["TMPDIR"])
digest = "a" * 64

def output(flag):
    if flag not in args:
        return None
    target = Path(args[args.index(flag) + 1])
    assert target.is_relative_to(root)
    return target

def save(target, value):
    if target is not None:
        target.write_text(json.dumps(value))

if args[0] == "sync":
    raise SystemExit(0)
if args[0] == "export":
    if scenario == "export-failure":
        raise SystemExit(19)
    target = output("--output-file")
    assert target is not None
    text = "safe_pkg==1.0 --hash=sha256:" + digest + "\n"
    if scenario == "empty-export":
        text = ""
    elif scenario == "bad-hashes":
        text = "safe_pkg==1.0\n"
    elif scenario == "conflicting-input":
        text += "Safe-Pkg==2.0 --hash=sha256:" + digest + "\n"
    elif scenario == "identical-input":
        text += "Safe.Pkg==1.0.0 --hash=sha256:" + digest + "\n"
    elif scenario in {"findings-partial", "findings-skipped"}:
        text += "missing_pkg==1.0 --hash=sha256:" + digest + "\n"
    elif scenario == "markers":
        text = 'Safe.Pkg==1.0; python_version >= "3.0" --hash=sha256:' + digest + "\n"
        text += 'ignored_pkg==9.0; python_version < "0" --hash=sha256:' + digest + "\n"
    elif scenario == "wrong-marker":
        text = 'safe_pkg==1.0; python_version < "0" --hash=sha256:' + digest + "\n"
        text += 'needed_pkg==1.0; python_version >= "3.0" --hash=sha256:' + digest + "\n"
    target.write_text(text)
    raise SystemExit(0)
assert args[0] == "run", args
rest = [part for part in args[1:] if part not in {"--no-sync", "--active", "--frozen"}]
tool = rest[0]
if tool == "python":
    assert rest[1] == "scripts/security_result.py"
    os.execv(sys.executable, [sys.executable, *rest[1:]])
if tool == "bandit":
    result = {"errors": [], "results": [], "metrics": {"_totals": {"loc": 100}}}
    code = 0
    if scenario == "bandit-missing-metrics":
        del result["metrics"]
    elif scenario == "bandit-zero-loc":
        result["metrics"]["_totals"]["loc"] = 0
    if scenario in {"bandit-error", "bandit-findings"}:
        result["results"] = [{"filename": "src/owned.py", "line_number": 3,
            "test_id": "B105", "issue_severity": "HIGH", "issue_confidence": "HIGH",
            "issue_text": "PRIVATE_LITERAL_MUST_NOT_APPEAR"}]
        code = 1
        if scenario == "bandit-error":
            result["errors"] = [{"reason": "controlled parse error"}]
    save(output("--output"), result)
    raise SystemExit(code)
if tool == "safety":
    # Baseline Make can be tested without contacting its obsolete service.
    raise SystemExit(1 if scenario in {"network-failure", "tool-failure"} else 0)
assert tool == "pip-audit", args
target = output("--output")
assert target is not None
if scenario == "network-failure":
    raise SystemExit(1)
if scenario == "tool-failure":
    raise SystemExit(7)
if scenario == "malformed":
    target.write_text("{")
    raise SystemExit(1)
row = {"name": "safe-pkg", "version": "1.0.0", "vulns": []}
finding = scenario in {"findings", "findings-partial", "findings-skipped", "exit0-findings", "repeated-vulnerability"}
if finding:
    row["vulns"] = [{"id": "CVE-2000-0001", "fix_versions": []}]
if scenario == "repeated-vulnerability":
    row["vulns"].extend([
        {"id": "CVE-2000-0001", "fix_versions": [], "description": "alternate advisory text"},
        {"id": "CVE-2000-0001", "fix_versions": [], "description": "third advisory row"},
    ])
result = {"dependencies": [row], "fixes": []}
code = 1 if finding else 0
if scenario == "partial":
    result["dependencies"] = []
elif scenario == "skipped":
    result["dependencies"] = [{"name": "safe-pkg", "skip_reason": "controlled"}]
    code = 1
elif scenario == "findings-skipped":
    result["dependencies"].append({"name": "missing-pkg", "skip_reason": "controlled"})
elif scenario == "wrong-name":
    row["name"] = "other-pkg"
elif scenario == "wrong-version":
    row["version"] = "2.0"
elif scenario == "duplicate":
    result["dependencies"].append(dict(row))
elif scenario == "fixes":
    result["fixes"] = [{"name": "safe-pkg"}]
elif scenario == "exit0-findings":
    code = 0
elif scenario == "exit1-clean":
    code = 1
save(target, result)
raise SystemExit(code)
"""


@dataclass(frozen=True)
class GateRun:
    process: subprocess.CompletedProcess[str]
    calls: list[list[str]]
    temporary: Path

    def outcome(self) -> dict[str, object]:
        rows = [
            json.loads(line) for line in self.process.stdout.splitlines() if line.startswith("{")
        ]
        assert rows, self.process.stdout + self.process.stderr
        return rows[-1]


def run_gate(
    tmp_path: Path, target: str, scenario: str, *, cleanup_failure: bool = False
) -> GateRun:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "Makefile", project / "Makefile")
    shutil.copyfile(HELPER, project / "scripts/security_result.py")
    (project / "pyproject.toml").write_text("[project]\nname='owned'\nversion='0'\n")
    (project / "uv.lock").write_text("version=1\n")
    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").symlink_to(sys.executable)
    temporary = tmp_path / "temporary"
    temporary.mkdir(mode=0o700)
    fake = tmp_path / "fake_uv.py"
    fake.write_text(FAKE_UV)
    calls = tmp_path / "calls.jsonl"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SAFETY_API_KEY", "MAKEFLAGS", "MFLAGS", "PYTHONPATH"}
    }
    env.update(
        SECURITY_TEST_LOG=str(calls),
        SECURITY_TEST_SCENARIO=scenario,
        TMPDIR=str(temporary),
        PYTHONDONTWRITEBYTECODE="1",
    )
    if cleanup_failure:
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        removal = fake_bin / "rm"
        removal.write_text("#!/bin/sh\nexit 1\n")
        removal.chmod(0o700)
        env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    command = [
        "make",
        "--no-print-directory",
        target,
        "UV=" + shlex.quote(sys.executable) + " " + shlex.quote(str(fake)),
        "VENV_DIR=" + str(environment),
    ]
    result = subprocess.run(
        command, cwd=project, env=env, capture_output=True, text=True, timeout=30
    )
    recorded = (
        [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    )
    return GateRun(result, recorded, temporary)


@pytest.mark.parametrize("target", ["sec", "sec-hard", "sec-soft"])
def test_make_keeps_locked_profile_and_explicit_auditor_contract(
    tmp_path: Path, target: str
) -> None:
    run = run_gate(tmp_path, target, "clean")
    assert run.process.returncode == 0, run.process.stdout + run.process.stderr
    assert run.calls[0][:2] == ["sync", "--frozen"]
    exported = next(row for row in run.calls if row[0] == "export")
    assert all(
        flag in exported for flag in ["--frozen", "--no-default-groups", "--no-emit-project"]
    )
    assert [exported[i + 1] for i, value in enumerate(exported) if value == "--extra"] == [
        "server",
        "mcp",
    ]
    assert all("--no-sync" in row for row in run.calls if row[0] == "run")
    audit = next(row for row in run.calls if "pip-audit" in row)
    assert all(
        flag in audit for flag in ["--require-hashes", "--no-deps", "--disable-pip", "--strict"]
    )
    assert "--fix" not in audit and "--ignore-vuln" not in audit
    assert run.outcome()["status"] == "clean"
    assert list(run.temporary.iterdir()) == []


@pytest.mark.parametrize(
    "target,success", [("sec", False), ("sec-hard", False), ("sec-soft", True)]
)
@pytest.mark.parametrize("scenario", ["findings", "bandit-findings"])
def test_findings_soften_only_in_soft_mode(
    tmp_path: Path, target: str, success: bool, scenario: str
) -> None:
    run = run_gate(tmp_path, target, scenario)
    assert (run.process.returncode == 0) is success
    assert run.outcome()["status"] == "findings"
    assert "PRIVATE_LITERAL_MUST_NOT_APPEAR" not in run.process.stdout
    assert list(run.temporary.iterdir()) == []


@pytest.mark.parametrize("target", ["sec-hard", "sec-soft"])
@pytest.mark.parametrize(
    "scenario",
    [
        "export-failure",
        "empty-export",
        "bad-hashes",
        "conflicting-input",
        "network-failure",
        "tool-failure",
        "malformed",
        "partial",
        "skipped",
        "wrong-name",
        "wrong-version",
        "duplicate",
        "exit0-findings",
        "exit1-clean",
        "findings-partial",
        "findings-skipped",
        "bandit-error",
        "bandit-missing-metrics",
        "bandit-zero-loc",
        "fixes",
        "wrong-marker",
    ],
)
def test_operational_or_incomplete_scans_never_soften(
    tmp_path: Path, target: str, scenario: str
) -> None:
    run = run_gate(tmp_path, target, scenario)
    assert run.process.returncode != 0, run.process.stdout + run.process.stderr
    if scenario in {"export-failure", "empty-export", "bad-hashes", "conflicting-input"}:
        assert not any(
            "bandit" in row or "pip-audit" in row or "safety" in row for row in run.calls
        )
    else:
        assert run.outcome()["status"] == "error"
    assert list(run.temporary.iterdir()) == []


@pytest.mark.parametrize("scenario", ["markers", "identical-input"])
def test_active_markers_and_exact_duplicate_pins_are_normalized(
    tmp_path: Path, scenario: str
) -> None:
    run = run_gate(tmp_path, "sec-hard", scenario)
    assert run.process.returncode == 0, run.process.stdout + run.process.stderr
    assert run.outcome()["status"] == "clean"
    assert list(run.temporary.iterdir()) == []


def test_cleanup_failure_is_not_a_success(tmp_path: Path) -> None:
    run = run_gate(tmp_path, "sec-soft", "clean", cleanup_failure=True)
    try:
        assert run.process.returncode != 0
        assert "Security scratch cleanup failed" in run.process.stderr
        assert list(run.temporary.iterdir())
    finally:
        shutil.rmtree(run.temporary)


@pytest.mark.parametrize(
    "target,success", [("sec", False), ("sec-hard", False), ("sec-soft", True)]
)
def test_repeated_vulnerability_ids_preserve_every_finding_row(
    tmp_path: Path, target: str, success: bool
) -> None:
    run = run_gate(tmp_path, target, "repeated-vulnerability")
    assert (run.process.returncode == 0) is success
    outcome = run.outcome()
    assert outcome["status"] == "findings"
    assert outcome["dependency_audit"]["status"] == "findings"
    assert outcome["dependency_audit"]["findings"] == 3
    assert len(outcome["dependency_report"]["dependencies"][0]["vulns"]) == 3
    assert list(run.temporary.iterdir()) == []
