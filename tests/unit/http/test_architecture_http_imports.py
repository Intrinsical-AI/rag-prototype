from __future__ import annotations

import ast
from pathlib import Path


def test_http_routers_do_not_import_infrastructure_directly() -> None:
    routers_dir = Path("src/local_rag_backend/app/routers")
    violations: list[str] = []
    disallowed_prefix = "local_rag_backend.infrastructure"

    for path in sorted(routers_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    f"{path}: import {alias.name}"
                    for alias in node.names
                    if alias.name.startswith(disallowed_prefix)
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(disallowed_prefix):
                    violations.append(f"{path}: from {module} import ...")

    assert not violations, "Direct infrastructure imports in HTTP routers:\\n" + "\\n".join(
        violations
    )
