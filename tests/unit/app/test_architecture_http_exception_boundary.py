from __future__ import annotations

from pathlib import Path


def test_http_exception_is_not_used_outside_app_http_package() -> None:
    app_root = Path("src/local_rag_backend/app")
    violations: list[str] = []

    for path in sorted(app_root.rglob("*.py")):
        if "http" in path.parts:
            continue
        content = path.read_text(encoding="utf-8")
        if "HTTPException" in content:
            violations.append(str(path))

    assert not violations, "HTTPException usage outside app/http:\\n" + "\\n".join(violations)
