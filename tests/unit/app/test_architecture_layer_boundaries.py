from __future__ import annotations

import ast
from pathlib import Path


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py")


def test_core_modules_do_not_import_app_or_infrastructure() -> None:
    core_root = Path("src/local_rag_backend/core")
    violations: list[str] = []
    disallowed_prefixes = (
        "local_rag_backend.app",
        "local_rag_backend.infrastructure",
    )

    for path in _iter_python_files(core_root):
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


def test_no_imports_point_to_removed_app_services_package() -> None:
    src_root = Path("src/local_rag_backend")
    violations: list[str] = []

    for path in _iter_python_files(src_root):
        content = path.read_text(encoding="utf-8")
        if "app.services" in content:
            violations.append(str(path))

    assert not violations, "Removed app.services package is still referenced:\n" + "\n".join(
        violations
    )


def test_removed_app_use_cases_directory_does_not_exist() -> None:
    assert not Path("src/local_rag_backend/app/use_cases").exists()
