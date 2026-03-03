from __future__ import annotations

import ast
from pathlib import Path


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py")


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
                if isinstance(node, ast.Import):
                    violations.extend(
                        f"{path}: import {alias.name}"
                        for alias in node.names
                        if alias.name.startswith(disallowed_prefixes)
                    )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.startswith(disallowed_prefixes):
                        violations.append(f"{path}: from {module} import ...")

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
