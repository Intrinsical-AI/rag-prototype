"""Private producer admission and installed-artifact rejection regressions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "smoke_canonical_wheels.py"
_spec = importlib.util.spec_from_file_location("canonical_wheel_smoke", SCRIPT)
assert _spec is not None and _spec.loader is not None
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def _wheel(
    root, name, *, version="0.5.0", metadata_name=None, dependency=None, suffix="", module=None
):
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


def _checksums(root):
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(root.glob("*.whl"))
        )
    )


@pytest.fixture
def wheelhouse(tmp_path):
    for name in smoke.TR3V0R_MODULES:
        _wheel(tmp_path, name)
    _checksums(tmp_path)
    return tmp_path


def test_five_distribution_receipt_binds_each_private_name_to_its_local_wheel(wheelhouse):
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
    wheelhouse, damage, refusal
):
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


def test_explicit_app_wheel_must_match_the_wheelhouse_receipt(wheelhouse, tmp_path):
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
    wheelhouse, tmp_path, producer, damage
):
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
