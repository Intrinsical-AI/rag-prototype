from __future__ import annotations

import ast
import importlib.util
import pkgutil
from pathlib import Path

import local_rag_backend


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _import_names(node: ast.Import | ast.ImportFrom, path: Path) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    package = ".".join(path.parent.relative_to("src").parts)
    module = (
        importlib.util.resolve_name("." * node.level + (node.module or ""), package)
        if node.level
        else node.module or ""
    )
    return [module, *(f"{module}.{alias.name}" for alias in node.names)]


def test_package_discovery_covers_every_source_module() -> None:
    root = Path("src/local_rag_backend")
    expected = set()
    for path in root.rglob("*.py"):
        parts = path.with_suffix("").relative_to("src").parts
        expected.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    discovered = {local_rag_backend.__name__} | {
        module.name
        for module in pkgutil.walk_packages(local_rag_backend.__path__, "local_rag_backend.")
    }
    assert expected <= discovered, f"Undiscovered source modules: {sorted(expected - discovered)}"


def test_relative_forbidden_imports_are_resolved() -> None:
    path = Path("src/local_rag_backend/core/services/example.py")
    for source in (
        "from ...infrastructure import persistence",
        "from ... import infrastructure",
    ):
        node = ast.parse(source).body[0]
        assert isinstance(node, ast.ImportFrom)
        assert any(
            name.startswith("local_rag_backend.infrastructure")
            for name in _import_names(node, path)
        )


def test_core_domain_and_services_do_not_import_infrastructure_or_http() -> None:
    """Domain, ports, and services layers must not import infrastructure or HTTP."""
    violations: list[str] = []
    disallowed_prefixes = (
        "local_rag_backend.infrastructure",
        "local_rag_backend.http",
        "local_rag_backend.composition",
    )

    for subdir in ("domain", "ports", "services"):
        root = Path(f"src/local_rag_backend/core/{subdir}")
        if not root.exists():
            continue
        for path in _iter_python_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    violations.extend(
                        f"{path}: import {name}"
                        for name in _import_names(node, path)
                        if name.startswith(disallowed_prefixes)
                    )

    assert not violations, "Core layer imports forbidden modules:\n" + "\n".join(violations)


def test_no_imports_point_to_removed_app_package() -> None:
    src_root = Path("src/local_rag_backend")
    violations: list[str] = []

    for path in _iter_python_files(src_root):
        content = path.read_text(encoding="utf-8")
        if "local_rag_backend.app." in content or "local_rag_backend.app " in content:
            violations.append(str(path))

    assert not violations, "Removed app package is still referenced:\n" + "\n".join(violations)


def test_removed_app_directory_does_not_exist() -> None:
    assert not Path("src/local_rag_backend/app").exists()
