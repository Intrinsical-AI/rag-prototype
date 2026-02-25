from __future__ import annotations

import ast
from pathlib import Path


def test_use_case_modules_do_not_import_http_transport_or_schemas() -> None:
    use_cases_dir = Path("src/local_rag_backend/core/use_cases")
    violations: list[str] = []

    disallowed_prefixes = (
        "fastapi",
        "starlette",
        "local_rag_backend.http",
    )

    for path in sorted(use_cases_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
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

    assert not violations, (
        "Use-case modules import transport/schema modules directly:\n" + "\n".join(violations)
    )


def test_use_case_modules_do_not_import_global_settings_singleton() -> None:
    use_cases_dir = Path("src/local_rag_backend/core/use_cases")
    violations: list[str] = []

    for path in sorted(use_cases_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    f"{path}: import {alias.name}"
                    for alias in node.names
                    if alias.name == "local_rag_backend.settings"
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module != "local_rag_backend.settings":
                    continue
                violations.extend(
                    f"{path}: from {node.module} import {alias.name}"
                    for alias in node.names
                    if alias.name == "settings"
                )

    assert not violations, (
        "Global settings singleton imports found in use-case modules:\n" + "\n".join(violations)
    )
