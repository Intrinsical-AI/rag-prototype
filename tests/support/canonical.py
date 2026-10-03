"""Current RepoGPT v5 producer payloads for transport boundary tests."""

from __future__ import annotations

import hashlib
from typing import Any


def repogpt_failure(path: str = "broken.py") -> dict[str, Any]:
    return {
        "record_type": "failure",
        "schema_version": "5",
        "path": path,
        "language": "py",
        "error": "parse failed",
        "file": {"size": 0, "sha256": hashlib.sha256(b"").hexdigest()},
    }


def repogpt_payload(
    *,
    external_id: str = "repogpt:demo:app.py:function:helper",
    content: str = "def helper():\n    return 1\n",
    failed_files: int = 0,
    failures: list[dict[str, Any]] | None = None,
    replace_scope: bool = True,
) -> dict[str, Any]:
    digest = hashlib.sha256(content.encode()).hexdigest()
    return {
        "kind": "code-units",
        "schema_version": "5",
        "repo_key": "demo",
        "scope": "repogpt:demo",
        "snapshot_id": "new",
        "replace_scope": replace_scope,
        "stats": {
            "total_files": 1 + failed_files,
            "ok_files": 1,
            "failed_files": failed_files,
            "emitted_documents": 1,
        },
        "failures": failures or [],
        "documents": [
            {
                "external_id": external_id,
                "source_id": "repogpt:demo:file:app.py",
                "repo_key": "demo",
                "scope": "repogpt:demo",
                "snapshot_id": "new",
                "path": "app.py",
                "language": "py",
                "unit_type": "function",
                "unit_level": "symbol",
                "symbol": "helper",
                "qualified_name": "helper",
                "container_id": "repogpt:demo:app.py:module",
                "depth": 1,
                "ancestor_path": ["app.py"],
                "start_line": 1,
                "end_line": len(content.splitlines()),
                "content": content,
                "content_hash": digest,
                "docstring_present": False,
                "has_children": False,
                "metadata": {
                    "file": {"sha256": digest, "size": len(content.encode())},
                    "tags": [],
                    "attributes": {},
                    "dependencies": [],
                },
            }
        ],
    }


def repogpt_empty_module_payload(
    *,
    failed_files: int = 0,
    failures: list[dict[str, Any]] | None = None,
    replace_scope: bool = True,
) -> dict[str, Any]:
    payload = repogpt_payload(
        failed_files=failed_files,
        failures=failures,
        replace_scope=replace_scope,
    )
    document = payload["documents"][0]
    module_id = "repogpt:demo:app.py:module"
    document.update(
        external_id=module_id,
        unit_type="module",
        unit_level="container",
        symbol=None,
        qualified_name="app.py",
        container_id=module_id,
        depth=0,
        ancestor_path=[],
        content="",
        content_ranges=[],
        content_hash=hashlib.sha256(b"").hexdigest(),
    )
    return payload
