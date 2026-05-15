from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from local_rag_backend import mcp_server
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.mcp_server import handle_request
from local_rag_backend.settings import settings


def _extract_content_text(response: dict[str, object]) -> dict[str, object]:
    result = response["result"]  # type: ignore[index]
    content = result["content"]  # type: ignore[index]
    return json.loads(content[0]["text"])  # type: ignore[index]


def test_mcp_initialize_and_list_tools() -> None:
    initialize = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert initialize["result"]["serverInfo"]["name"] == "rag-prototype"  # type: ignore[index]

    listed = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = {tool["name"] for tool in listed["result"]["tools"]}  # type: ignore[index]
    assert {
        "rag_ask",
        "rag_status",
        "rag_import_canonical",
        "rag_rebuild_index",
        "rag_eval",
    } <= names


def test_mcp_ask_uses_configured_rag_service(monkeypatch) -> None:
    calls = []

    class FakeRagService:
        def ask(self, question, top_k, filters, retrieval_mode):
            calls.append((question, top_k, filters, retrieval_mode))
            return {
                "answer": "runtime answer",
                "docs": [
                    Document(
                        id="doc:1",
                        content="context",
                        external_id="ext-1",
                        source_id="source-1",
                        metadata={"scope": "demo"},
                    )
                ],
                "scores": [0.75],
            }

    class FakeContainer:
        settings_obj = SimpleNamespace(retrieval_mode="sparse")

        def build_rag_service(self):
            return FakeRagService()

    monkeypatch.setattr(mcp_server, "get_cli_container", lambda: FakeContainer())

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "rag_ask",
                "arguments": {
                    "question": "  what is indexed?  ",
                    "k": 1,
                    "filters": [{"field": "metadata.scope", "values": ["demo"]}],
                },
            },
        }
    )

    payload = _extract_content_text(response)
    assert payload == {
        "answer": "runtime answer",
        "sources": [
            {
                "document": {
                    "id": "doc:1",
                    "content": "context",
                    "external_id": "ext-1",
                    "source_id": "source-1",
                    "metadata": {"scope": "demo"},
                },
                "score": 0.75,
            }
        ],
    }
    assert calls[0][0] == "what is indexed?"
    assert calls[0][1] == 1
    assert calls[0][2][0].field == "metadata.scope"
    assert calls[0][3] == "sparse"


def test_mcp_ask_rejects_invalid_k() -> None:
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "rag_ask",
                "arguments": {"question": "hello", "k": 0},
            },
        }
    )

    assert "error" in response
    assert "k must be between 1 and 10" in response["error"]["message"]  # type: ignore[index]


def test_mcp_import_canonical_and_status(in_memory_sqlite, tmp_path, monkeypatch) -> None:
    _ = in_memory_sqlite
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data", raising=False)
    settings.data_dir.mkdir()

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "rag_import_canonical",
                "arguments": {
                    "payload": {
                        "scope": "repogpt:demo",
                        "snapshot_id": "snap-1",
                        "documents": [
                            {
                                "external_id": "doc-1",
                                "source_id": "repogpt:demo:file:src/app.py",
                                "content": "def helper():\n    return 1\n",
                                "metadata": {"path": "src/app.py", "unit_type": "function"},
                            }
                        ],
                    }
                },
            },
        }
    )

    payload = _extract_content_text(response)
    assert payload["inserted"] == 1
    assert payload["replace_scope"] is True
    assert {doc.external_id for doc in SqlDocumentStorage().get_all_documents()} == {"doc-1"}

    status = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "rag_status", "arguments": {}},
        }
    )
    status_payload = _extract_content_text(status)
    assert status_payload["runtime"]["backends"]["retrieval"] == "sparse"
    assert status_payload["health"]["documents_count"] == 1
    assert status_payload["health"]["history_count"] == 0


def test_mcp_import_canonical_rejects_repogpt_schema_v3(
    in_memory_sqlite, tmp_path, monkeypatch
) -> None:
    _ = (in_memory_sqlite, tmp_path)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "rag_import_canonical",
                "arguments": {
                    "payload": {
                        "schema_version": "3",
                        "kind": "code-units",
                        "repo_key": "demo",
                        "scope": "repogpt:demo",
                        "snapshot_id": "snap-1",
                        "documents": [
                            {
                                "external_id": "repogpt:demo:src/app.py:function:helper",
                                "source_id": "repogpt:demo:file:src/app.py",
                                "scope": "repogpt:demo",
                                "snapshot_id": "snap-1",
                                "path": "src/app.py",
                                "unit_type": "function",
                                "repo_key": "demo",
                                "content_hash": "abc123",
                                "content": "def helper():\n    return 1\n",
                                "metadata": {
                                    "scope": "repogpt:demo",
                                    "snapshot_id": "snap-1",
                                    "path": "src/app.py",
                                    "unit_type": "function",
                                    "repo_key": "demo",
                                    "content_hash": "abc123",
                                },
                            }
                        ],
                    }
                },
            },
        }
    )

    assert "error" in response
    assert "schema_version='4'" in response["error"]["message"]  # type: ignore[index]


def test_mcp_eval_supports_filters(in_memory_sqlite, tmp_path, monkeypatch) -> None:
    _ = in_memory_sqlite
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "search_backend", "local_split", raising=False)
    monkeypatch.setattr(settings, "persistence_backend", "local_split", raising=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data", raising=False)
    settings.data_dir.mkdir()

    dataset_path = Path(__file__).resolve().parents[2] / "datasets" / "repogpt_rag_eval_v1.jsonl"
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "rag_eval",
                "arguments": {
                    "dataset_path": str(dataset_path),
                    "retrieval_mode": "sparse",
                    "k": 1,
                    "filters": [{"field": "source_id", "values": ["eval:repogpt:v1"]}],
                },
            },
        }
    )

    payload = _extract_content_text(response)
    assert payload["dataset_id"] == "repogpt_rag_eval_v1"
    assert payload["k"] == 1
    assert payload["queries"] > 0
    assert set(payload["metrics"]) == {"nDCG@1", "MAP@1", "MRR@1", "P@1", "Recall@1"}
    assert payload["filters"] == [{"field": "source_id", "values": ["eval:repogpt:v1"]}]
