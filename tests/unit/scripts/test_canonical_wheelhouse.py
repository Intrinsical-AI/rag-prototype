"""Private producer admission and installed-artifact rejection regressions."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "smoke_canonical_wheels.py"
_spec = importlib.util.spec_from_file_location("canonical_wheel_smoke", SCRIPT)
assert _spec is not None and _spec.loader is not None
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def _wheel(
    root: Path,
    name: str,
    *,
    version: str = "0.5.0",
    metadata_name: str | None = None,
    dependency: str | None = None,
    suffix: str = "",
    module: str | None = None,
) -> Path:
    stem = name.replace("-", "_")
    path = root / f"{stem}-{version}{suffix}-py3-none-any.whl"
    metadata = f"Name: {metadata_name or name}\nVersion: {version}\n"
    if dependency:
        metadata += f"Requires-Dist: {dependency}\n"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            f"{module or smoke.TR3V0R_MODULES[name]}/__init__.py", "# synthetic wheel\n"
        )
        archive.writestr(f"{stem}-{version}.dist-info/METADATA", metadata)
    return path


def _checksums(root: Path) -> None:
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(root.glob("*.whl"))
        )
    )


@pytest.fixture
def wheelhouse(tmp_path: Path) -> Path:
    for name in smoke.TR3V0R_MODULES:
        _wheel(tmp_path, name)
    _checksums(tmp_path)
    return tmp_path


def test_five_distribution_receipt_binds_each_private_name_to_its_local_wheel(
    wheelhouse: Path,
) -> None:
    wheels = smoke._tr3v0r_wheels(wheelhouse / "tr3v0r-0.5.0-py3-none-any.whl", wheelhouse)
    assert set(wheels) == set(smoke.TR3V0R_MODULES)
    constraints = smoke._private_constraints(wheelhouse, wheels).read_text().splitlines()
    assert set(constraints) == {
        f"{name} @ {Path(artifact['path']).as_uri()}" for name, artifact in wheels.items()
    }


@pytest.mark.parametrize(
    "damage,refusal",
    [
        ("missing", "all five"),
        ("duplicate", "duplicate"),
        ("wrong_distribution", "Expected tr3v0r-core"),
        ("version", "versions must match"),
        ("checksum", "checksum mismatch"),
        ("unbound_private_dependency", "Unbound private producer"),
    ],
)
def test_invalid_private_wheelhouse_is_rejected_before_dependency_resolution(
    wheelhouse: Path, damage: str, refusal: str
) -> None:
    core = wheelhouse / "tr3v0r_core-0.5.0-py3-none-any.whl"
    if damage == "missing":
        core.unlink()
    elif damage == "duplicate":
        _wheel(wheelhouse, "tr3v0r-core", suffix="-1")
    elif damage == "wrong_distribution":
        _wheel(wheelhouse, "tr3v0r-core", metadata_name="unrelated")
    elif damage == "version":
        core.unlink()
        _wheel(wheelhouse, "tr3v0r-core", version="0.4.0")
    elif damage == "unbound_private_dependency":
        _wheel(wheelhouse, "tr3v0r-core", dependency="tr3v0r-private-legacy>=1")
    else:
        with core.open("ab") as stream:
            stream.write(b"unreceipted bytes")
    if damage != "checksum":
        _checksums(wheelhouse)
    with pytest.raises(ValueError, match=refusal):
        smoke._tr3v0r_wheels(wheelhouse / "tr3v0r-0.5.0-py3-none-any.whl", wheelhouse)


def test_explicit_app_wheel_must_match_the_wheelhouse_receipt(
    wheelhouse: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    app = _wheel(outside, "tr3v0r", version="0.4.0")
    with pytest.raises(ValueError, match="app wheel does not match"):
        smoke._tr3v0r_wheels(app, wheelhouse)


@pytest.mark.parametrize(
    "producer,damage",
    [
        ("tr3v0r", damage)
        for damage in [None, "bytes", "origin", "direct_url", "version", "input_hash"]
    ]
    + [
        ("repogpt", "bytes"),
        ("chat-exports", "bytes"),
        ("repogpt", "direct_url"),
        ("chat-exports", "direct_url"),
    ],
)
def test_installed_verifier_checks_bytes_origin_direct_url_version_and_input_hash(
    wheelhouse: Path, tmp_path: Path, producer: str, damage: str | None
) -> None:
    environment = tmp_path / "installed"
    environment.mkdir()
    module_name = {"tr3v0r": "tr3v0r_app", "repogpt": "repogpt", "chat-exports": "chat_exports"}[
        producer
    ]
    app = _wheel(wheelhouse, producer, module=module_name)
    expected = smoke._wheel(app, producer)
    with ZipFile(app) as archive:
        archive.extractall(environment)
    dist_info = environment / f"{producer.replace('-', '_')}-0.5.0.dist-info"
    direct_url = dist_info / "direct_url.json"
    direct_url.write_text(json.dumps({"url": app.resolve().as_uri()}))
    module = environment / module_name / "__init__.py"
    injection = ""
    if damage == "bytes":
        module.write_text("# changed installed bytes\n")
    elif damage == "origin":
        external = tmp_path / "external"
        (external / module_name).mkdir(parents=True)
        (external / module_name / "__init__.py").write_text("# outside environment\n")
        injection = f"sys.path.insert(0, {str(external)!r});"
    elif damage == "direct_url":
        direct_url.write_text(json.dumps({"url": "file:///wrong-wheel.whl"}))
    elif damage == "version":
        (dist_info / "METADATA").write_text(f"Name: {producer}\nVersion: 0.4.0\n")
    elif damage == "input_hash":
        expected["sha256"] = "0" * 64
    setup = (
        f"import sys;sys.prefix={str(environment)!r};sys.path.insert(0,sys.prefix);{injection}\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            setup + smoke.VERIFY_INSTALLED,
            json.dumps({producer: expected}),
            json.dumps({producer: module_name}),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if damage is None:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)[producer]["wheel_sha256"] == expected["sha256"]
    else:
        assert result.returncode != 0
        assert "AssertionError" in result.stderr


class _BudgetClock:
    def __init__(self) -> None:
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def advance(self, seconds: float) -> None:
        self.elapsed += seconds


def _generated_runner(
    text: str,
    clock: _BudgetClock,
    diagnostics: io.StringIO,
    *,
    seconds: float,
    returncode: int = 0,
    partial_stdout: str | bytes = "",
    partial_stderr: str | bytes = "",
) -> tuple[Callable[..., subprocess.CompletedProcess[str]], list[float]]:
    """Compile the actual generated runner definitions, never imports or app body."""
    tree = ast.parse(text)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert any(node.name == "run" for node in definitions)
    observed_timeouts: list[float] = []

    def fake_step(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        timeout = float(kwargs["timeout"])
        observed_timeouts.append(timeout)
        if seconds > timeout:
            clock.advance(timeout)
            raise subprocess.TimeoutExpired(
                command, timeout, output=partial_stdout, stderr=partial_stderr
            )
        clock.advance(seconds)
        return subprocess.CompletedProcess(
            command, returncode, "controlled stdout", "controlled stderr"
        )

    namespace: dict[str, Any] = {
        "__name__": "controlled_generated_runner",
        "json": json,
        "sys": SimpleNamespace(stderr=diagnostics, stdout=io.StringIO()),
        "time": SimpleNamespace(monotonic=clock.monotonic, perf_counter=clock.monotonic),
        "monotonic": clock.monotonic,
        "perf_counter": clock.monotonic,
        "Path": Path,
        "Any": Any,
        "subprocess": SimpleNamespace(
            run=fake_step,
            CompletedProcess=subprocess.CompletedProcess,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    }
    # Carry literal runner constants from the exact generated text; do not
    # evaluate any application/configuration/fixture assignment or import.
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(value, (str, int, float, bool, type(None))):
                for target in node.targets:
                    assert isinstance(target, ast.Name)
                    namespace[target.id] = value
    module = ast.fix_missing_locations(ast.Module(body=[*definitions], type_ignores=[]))
    # Intentional test: only AST-selected FunctionDef nodes and literal-seeded
    # globals execute; imports and the application body are excluded.
    exec(compile(module, "<actual-generated-consumer-runner>", "exec"), namespace)  # noqa: S102
    return cast(
        Callable[..., subprocess.CompletedProcess[str]], namespace["run"]
    ), observed_timeouts


def _fake_canonical_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    seconds_per_step: float,
    force_outer_timeout: bool = False,
    consumer_returncode: int = 0,
) -> dict[str, Any]:
    """Fake every resolver/admission/source/RF collaborator; exercise real main/_run."""
    scratch = tmp_path / "owned-main-scratch"
    scratch.mkdir()
    trace: dict[str, Any] = {"outer_calls": [], "inner_timeouts": [], "generated": None}

    def fake_wheel(path: Path, name: str) -> dict[str, str]:
        version = {"repogpt": "0.12.1", "chat-exports": "0.4.1", "rag-prototype": "5.0.1"}.get(
            name, "0.5.0"
        )
        return {"path": str(path), "version": version, "sha256": "a" * 64}

    def fake_tr3(app: Path, wheelhouse: Path) -> dict[str, dict[str, str]]:
        return {name: fake_wheel(wheelhouse / f"{name}.whl", name) for name in smoke.TR3V0R_MODULES}

    def fake_source(_environment: dict[str, str]) -> dict[str, Any]:
        return {
            "revision": "controlled-source",
            "tree": "controlled-tree",
            "dirty": False,
            "files": {},
        }

    def fake_constraints(root: Path, _wheels: dict[str, Any]) -> Path:
        path = root / "controlled-constraints.txt"
        path.write_text("# no resolution or installation in this test\n")
        return path

    def fake_outer(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        timeout = float(kwargs["timeout"])
        combined = len(command) > 1 and Path(command[1]).name == "consumer.py"
        trace["outer_calls"].append(
            {"command": list(command), "timeout": timeout, "combined": combined}
        )
        if not combined:
            # All UV/verification commands are fakes; the installed verifier
            # response is deliberately not evidence of a real installation.
            return subprocess.CompletedProcess(command, 0, "{}", "")
        consumer = Path(command[1])
        assert consumer.is_relative_to(scratch)
        text = consumer.read_text()
        trace["generated"] = text
        clock = _BudgetClock()
        diagnostics = io.StringIO()
        runner, timeouts = _generated_runner(text, clock, diagnostics, seconds=seconds_per_step)
        for index in range(15):
            runner(["controlled-cli", str(index)], expected=0)
        trace["inner_timeouts"] = timeouts
        trace["aggregate_seconds"] = clock.elapsed
        stderr = (
            diagnostics.getvalue()
            + json.dumps({"event": "start", "label": "final-checks", "elapsed_seconds": 0})
            + "\nFAKE-CONSUMER-STDERR-TAIL\n"
        )
        if force_outer_timeout:
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=b"x" * 64_000 + b"\xffSTDOUT-PARTIAL-TAIL",
                stderr=b"y" * 64_000 + b"\xffLAST-STEP-PARTIAL-TAIL",
            )
        if clock.elapsed > timeout:
            raise subprocess.TimeoutExpired(command, timeout, output="", stderr=stderr)
        return subprocess.CompletedProcess(
            command, consumer_returncode, json.dumps({"controlled_boundary": True}), stderr
        )

    monkeypatch.setattr(smoke, "_wheel", fake_wheel)
    monkeypatch.setattr(smoke, "_tr3v0r_wheels", fake_tr3)
    monkeypatch.setattr(smoke, "_consumer_source", fake_source)
    monkeypatch.setattr(smoke, "_verify_source_wheel", lambda *_args: None)
    monkeypatch.setattr(smoke, "_private_constraints", fake_constraints)
    monkeypatch.setattr(
        smoke,
        "_isolated_producer_boundary",
        lambda *_args, **_kwargs: {"controlled_isolated": True},
    )
    monkeypatch.setattr(smoke, "shutil", SimpleNamespace(which=lambda _name: "/controlled/uv"))
    monkeypatch.setattr(smoke, "os", SimpleNamespace(environ={}))
    monkeypatch.setattr(
        smoke,
        "subprocess",
        SimpleNamespace(
            run=fake_outer,
            TimeoutExpired=subprocess.TimeoutExpired,
            CompletedProcess=subprocess.CompletedProcess,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke-canonical-wheels",
            "--repogpt-wheel",
            str(tmp_path / "repogpt.whl"),
            "--chat-exports-wheel",
            str(tmp_path / "chat.whl"),
            "--rag-wheel",
            str(tmp_path / "rag.whl"),
            "--tr3v0r-wheel",
            str(tmp_path / "tr3.whl"),
            "--tr3v0r-wheelhouse",
            str(tmp_path / "wheelhouse"),
            "--tr3v0r-example-script",
            str(tmp_path / "generator.py"),
            "--scratch-root",
            str(scratch),
        ],
    )
    return trace


def _diagnostic_events(raw: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in raw.splitlines() if line.startswith("{")]


def test_combined_budget_allows_fifteen_valid_steps_beyond_two_minutes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace = _fake_canonical_main(monkeypatch, tmp_path, seconds_per_step=9)
    smoke.main()
    assert trace["aggregate_seconds"] == 135
    assert trace["inner_timeouts"] == [60] * 15
    combined = [row for row in trace["outer_calls"] if row["combined"]]
    assert len(combined) == 1 and combined[0]["timeout"] == 15 * 60 + 120
    assert all(row["timeout"] == 120 for row in trace["outer_calls"] if not row["combined"])
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_combined_success_preserves_stderr_and_final_stdout_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_canonical_main(monkeypatch, tmp_path, seconds_per_step=1)
    smoke.main()
    captured = capsys.readouterr()
    assert "FAKE-CONSUMER-STDERR-TAIL" in captured.err
    assert "FAKE-CONSUMER-STDERR-TAIL" not in captured.out
    assert json.loads(captured.out)["boundary"] == {"controlled_boundary": True}
    terminal = _diagnostic_events(captured.err)[-1]
    assert terminal["last_step"]["label"] == "final-checks"
    assert terminal["last_step"]["event"] == "start"


def test_combined_outer_timeout_preserves_bounded_partial_byte_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_canonical_main(monkeypatch, tmp_path, seconds_per_step=1, force_outer_timeout=True)
    with pytest.raises(subprocess.TimeoutExpired):
        smoke.main()
    captured = capsys.readouterr()
    assert "STDOUT-PARTIAL-TAIL" in captured.err
    assert "LAST-STEP-PARTIAL-TAIL" in captured.err
    assert len(captured.err.encode("utf-8")) <= 16_384
    assert captured.out == ""


def test_combined_nonzero_error_preserves_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_canonical_main(monkeypatch, tmp_path, seconds_per_step=1, consumer_returncode=7)
    with pytest.raises(RuntimeError):
        smoke.main()
    captured = capsys.readouterr()
    assert "FAKE-CONSUMER-STDERR-TAIL" in captured.err
    assert captured.out == ""


def test_generated_step_keeps_sixty_second_timeout() -> None:
    runner, timeouts = _generated_runner(smoke.CONSUMER, _BudgetClock(), io.StringIO(), seconds=61)
    with pytest.raises(subprocess.TimeoutExpired):
        runner(["controlled-slow-cli"], expected=0)
    assert timeouts == [60]


@pytest.mark.parametrize("expected", [1, 2])
def test_generated_step_keeps_expected_nonzero_exit_control(expected: int) -> None:
    runner, timeouts = _generated_runner(
        smoke.CONSUMER, _BudgetClock(), io.StringIO(), seconds=1, returncode=expected
    )
    assert runner(["controlled-expected-exit"], expected=expected).returncode == expected
    assert timeouts == [60]


@pytest.mark.parametrize("expected", [1, 2])
def test_generated_step_accepts_expected_nonzero_exit_and_emits_phases(expected: int) -> None:
    diagnostics = io.StringIO()
    runner, _timeouts = _generated_runner(
        smoke.CONSUMER, _BudgetClock(), diagnostics, seconds=1, returncode=expected
    )
    assert runner(["controlled-expected-exit"], expected=expected).returncode == expected
    events = _diagnostic_events(diagnostics.getvalue())
    assert [event["event"] for event in events] == ["start", "end"]
    assert all(isinstance(event["label"], str) and event["label"] for event in events)
    assert events[-1]["returncode"] == expected and events[-1]["elapsed_seconds"] == 1


def test_generated_step_refuses_unexpected_nonzero_exit() -> None:
    runner, _timeouts = _generated_runner(
        smoke.CONSUMER, _BudgetClock(), io.StringIO(), seconds=1, returncode=7
    )
    with pytest.raises(AssertionError):
        runner(["controlled-wrong-exit"], expected=0)


def test_generated_wrong_exit_emits_terminal_diagnostic_before_refusal() -> None:
    diagnostics = io.StringIO()
    runner, _timeouts = _generated_runner(
        smoke.CONSUMER, _BudgetClock(), diagnostics, seconds=1, returncode=7
    )
    with pytest.raises(AssertionError):
        runner(["controlled-wrong-exit"], expected=0)
    events = _diagnostic_events(diagnostics.getvalue())
    assert [event["event"] for event in events] == ["start", "end"]
    assert events[-1]["returncode"] == 7 and events[-1]["elapsed_seconds"] == 1


def test_generated_timeout_emits_bounded_partial_byte_tails() -> None:
    diagnostics = io.StringIO()
    runner, _timeouts = _generated_runner(
        smoke.CONSUMER,
        _BudgetClock(),
        diagnostics,
        seconds=61,
        partial_stdout=b"x" * 64_000 + b"\xffSTDOUT-PARTIAL-TAIL",
        partial_stderr=b"y" * 64_000 + b"\xffSTDERR-PARTIAL-TAIL",
    )
    with pytest.raises(subprocess.TimeoutExpired):
        runner(["controlled-timeout"], expected=0)
    events = _diagnostic_events(diagnostics.getvalue())
    assert [event["event"] for event in events] == ["start", "timeout"]
    assert events[-1]["elapsed_seconds"] == 60
    for field, marker in (
        ("stdout_tail", "STDOUT-PARTIAL-TAIL"),
        ("stderr_tail", "STDERR-PARTIAL-TAIL"),
    ):
        assert marker in events[-1][field]
        assert len(events[-1][field].encode("utf-8")) <= 4096
    assert len(diagnostics.getvalue().encode("utf-8")) <= 16_384


def test_default_nonconsumer_command_budget_remains_two_minutes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[float] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.append(float(kwargs["timeout"]))
        return subprocess.CompletedProcess(command, 0, "controlled result", "")

    monkeypatch.setattr(
        smoke, "subprocess", SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired)
    )
    assert smoke._run(["/controlled/uv", "--version"], cwd=tmp_path, env={}) == "controlled result"
    assert observed == [120]
