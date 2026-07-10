from __future__ import annotations

import ast
from pathlib import Path


def test_http_routers_do_not_import_global_settings_singleton() -> None:
    routers_dir = Path("src/local_rag_backend/http/routers")
    violations: list[str] = []
    settings_module = "local_rag_backend.settings"

    for path in sorted(routers_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    f"{path}: import {alias.name}"
                    for alias in node.names
                    if alias.name == settings_module
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module != settings_module:
                    continue
                violations.extend(
                    f"{path}: from {node.module} import {alias.name}"
                    for alias in node.names
                    if alias.name == "settings"
                )

    assert not violations, "Global settings singleton imports found in HTTP routers:\n" + "\n".join(
        violations
    )
