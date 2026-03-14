from __future__ import annotations

import ast
from pathlib import Path


def test_http_routers_do_not_import_infrastructure_adapters_directly() -> None:
    """Routers must not import persistence, retrieval, LLM, or embedding adapters.

    Cross-cutting infrastructure (concurrency, observability) is allowed — these
    are operational utilities, not domain adapters behind ports.
    """
    routers_dir = Path("src/local_rag_backend/http/routers")
    violations: list[str] = []
    disallowed_prefixes = (
        "local_rag_backend.infrastructure.persistence",
        "local_rag_backend.infrastructure.retrieval",
        "local_rag_backend.infrastructure.llms",
        "local_rag_backend.infrastructure.embeddings",
        "local_rag_backend.infrastructure.ingestion",
    )

    for path in sorted(routers_dir.glob("*.py")):
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

    assert not violations, "Direct infrastructure adapter imports in HTTP routers:\n" + "\n".join(
        violations
    )
