from __future__ import annotations

import ast
from pathlib import Path

# Intentional zero-debt baseline: any new infra/composition import in use cases must be explicit.
_EXPECTED_USE_CASE_INFRA_IMPORTS: dict[str, set[str]] = {}

_DEBT_PREFIXES = (
    "local_rag_backend.infrastructure",
    "local_rag_backend.composition",
)


def _iter_use_case_files() -> list[Path]:
    root = Path("src/local_rag_backend/core/use_cases")
    return sorted(path for path in root.glob("*.py") if path.name != "__init__.py")


def _collect_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _collect_actual_use_case_debt() -> dict[str, set[str]]:
    actual: dict[str, set[str]] = {}
    for path in _iter_use_case_files():
        imports = _collect_modules(path)
        debt = {module for module in imports if module.startswith(_DEBT_PREFIXES)}
        if debt:
            actual[str(path)] = debt
    return actual


def _format_mapping_diff(prefix: str, mapping: dict[str, set[str]]) -> list[str]:
    lines: list[str] = []
    for file_path in sorted(mapping):
        modules = ", ".join(sorted(mapping[file_path]))
        lines.append(f"{prefix} {file_path} -> {modules}")
    return lines


def test_frozen_use_case_infrastructure_debt_snapshot() -> None:
    actual = _collect_actual_use_case_debt()
    expected = _EXPECTED_USE_CASE_INFRA_IMPORTS

    new_files = sorted(set(actual) - set(expected))
    removed_files = sorted(set(expected) - set(actual))

    new_modules: dict[str, set[str]] = {}
    removed_modules: dict[str, set[str]] = {}
    for file_path in sorted(set(actual) & set(expected)):
        added = actual[file_path] - expected[file_path]
        removed = expected[file_path] - actual[file_path]
        if added:
            new_modules[file_path] = added
        if removed:
            removed_modules[file_path] = removed

    if not (new_files or removed_files or new_modules or removed_modules):
        return

    lines = ["Use-case infra/composition debt snapshot changed."]
    if new_files:
        lines.extend(f"NEW FILE DEBT {file_path}" for file_path in new_files)
    if removed_files:
        lines.extend(f"REMOVED FILE DEBT {file_path}" for file_path in removed_files)
    lines.extend(_format_mapping_diff("NEW IMPORT DEBT", new_modules))
    lines.extend(_format_mapping_diff("REMOVED IMPORT DEBT", removed_modules))
    lines.append("If intentional, update this snapshot.")
    lines.append("si removiste deuda, actualiza snapshot")
    raise AssertionError("\n".join(lines))
