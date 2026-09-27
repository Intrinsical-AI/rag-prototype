import tomllib
from importlib.metadata import version
from pathlib import Path


def test_package_version_matches_dist_metadata():
    import local_rag_backend

    assert local_rag_backend.__version__ == version("rag-prototype")
    assert local_rag_backend.__version__ != "0.0.0"


def test_installed_version_matches_project_and_lock() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    packages = tomllib.loads((root / "uv.lock").read_text())["package"]
    locked = [package["version"] for package in packages if package["name"] == project["name"]]
    assert locked == [project["version"]]
    assert version(project["name"]) == project["version"]
