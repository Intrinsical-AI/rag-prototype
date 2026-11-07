"""Distribution admission binds documentation and metadata to declared source inputs."""

from __future__ import annotations

import configparser
import hashlib
import importlib.util
import io
import tarfile
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "own_package_smoke", ROOT / "scripts/smoke_package.py"
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _pair(
    tmp_path: Path, *, stale_readme: bool = False, changed_sdist: bool = False
) -> tuple[Path, Path, dict[str, str], dict[str, bytes]]:
    inputs = {
        "README.md": b"# Current synthetic package documentation\n",
        "LICENSE": (ROOT / "LICENSE").read_bytes(),
        "pyproject.toml": (ROOT / "pyproject.toml").read_bytes(),
    }
    project = tomllib.loads(inputs["pyproject.toml"].decode())["project"]
    version = project["version"]
    runtime_name = "local_rag_backend/__init__.py"
    runtime = b"# Generated fixture; never imported or executed.\n"
    package = {"src/" + runtime_name: hashlib.sha256(runtime).hexdigest()}

    def metadata(readme: bytes) -> bytes:
        fields = [
            "Metadata-Version: 2.4",
            f"Name: {project['name']}",
            f"Version: {version}",
            f"Requires-Python: {project['requires-python']}",
            f"License-Expression: {project['license']}",
            "License-File: LICENSE",
        ]
        fields.extend("Provides-Extra: " + extra for extra in project["optional-dependencies"])
        fields.extend("Requires-Dist: " + item for item in project["dependencies"])
        for extra, dependencies in project["optional-dependencies"].items():
            fields.extend(f'Requires-Dist: {item}; extra == "{extra}"' for item in dependencies)
        return ("\n".join(fields) + "\n\n").encode() + readme

    entrypoints = configparser.ConfigParser()
    entrypoints["console_scripts"] = project["scripts"]
    with io.StringIO() as stream:
        entrypoints.write(stream)
        entrypoints_raw = stream.getvalue().encode()
    dist_info = f"rag_prototype-{version}.dist-info/"
    wheel = tmp_path / f"rag_prototype-{version}-py3-none-any.whl"
    wheel_readme = b"# Retired synthetic documentation\n" if stale_readme else inputs["README.md"]
    with ZipFile(wheel, "w") as archive:
        archive.writestr(runtime_name, runtime)
        archive.writestr(dist_info + "METADATA", metadata(wheel_readme))
        archive.writestr(dist_info + "entry_points.txt", entrypoints_raw)
        archive.writestr(dist_info + "licenses/LICENSE", inputs["LICENSE"])
        archive.writestr(dist_info + "WHEEL", "Wheel-Version: 1.0\nGenerator: synthetic-fixture\n")

    source_files = {
        **inputs,
        "src/" + runtime_name: runtime,
        "PKG-INFO": metadata(inputs["README.md"]),
        "src/rag_prototype.egg-info/PKG-INFO": metadata(inputs["README.md"]),
        "src/rag_prototype.egg-info/entry_points.txt": entrypoints_raw,
    }
    if changed_sdist:
        source_files["pyproject.toml"] += b"\n# Unreviewed source change\n"
    sdist = tmp_path / f"rag_prototype-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, content in source_files.items():
            member = tarfile.TarInfo(f"rag_prototype-{version}/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return wheel, sdist, package, inputs


def test_current_declared_package_pair_is_accepted(tmp_path: Path) -> None:
    smoke.check_pair(*_pair(tmp_path))


def test_same_runtime_wheel_with_stale_readme_is_rejected(tmp_path: Path) -> None:
    wheel, sdist, package, inputs = _pair(tmp_path, stale_readme=True)
    with ZipFile(wheel) as archive:
        assert (
            hashlib.sha256(archive.read("local_rag_backend/__init__.py")).hexdigest()
            == (package["src/local_rag_backend/__init__.py"])
        )
    with pytest.raises(AssertionError, match="README metadata differs from source"):
        smoke.check_pair(wheel, sdist, package, inputs)


def test_sdist_with_unreviewed_source_metadata_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="Sdist source input differs from Git"):
        smoke.check_pair(*_pair(tmp_path, changed_sdist=True))
