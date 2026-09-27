from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_rag_backend import mcp_server
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.mcp_server import handle_request
from local_rag_backend.settings import get_settings

settings = get_settings()


def _extract_content_text(response: dict[str, object]) -> dict[str, object]:
    result = response["result"]  # type: ignore[index]
    content = result["content"]  # type: ignore[index]
    return json.loads(content[0]["text"])  # type: ignore[index]


def test_mcp_initialize_and_list_tools() -> None:
    initialize = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "rag-prototype"  # type: ignore[index]
    assert initialize["result"]["protocolVersion"] == "2024-11-05"

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
    assert response["error"]["code"] == -32602
    assert "k" in response["error"]["message"]


@pytest.mark.parametrize("invalid_k", [True, "1", 1.0])
def test_mcp_ask_rejects_non_integer_k(invalid_k) -> None:
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {
                "name": "rag_ask",
                "arguments": {"question": "hello", "k": invalid_k},
            },
        }
    )

    assert "error" in response
    assert response["error"]["code"] == -32602
    assert "integer" in response["error"]["message"]


def test_mcp_ask_rejects_question_over_schema_limit() -> None:
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 23,
            "method": "tools/call",
            "params": {
                "name": "rag_ask",
                "arguments": {"question": "x" * (mcp_server.MAX_ASK_QUESTION_CHARS + 1)},
            },
        }
    )

    assert "error" in response
    assert response["error"]["code"] == -32602
    assert "4096" in response["error"]["message"]


def test_mcp_ask_rejects_non_string_question() -> None:
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 24,
            "method": "tools/call",
            "params": {"name": "rag_ask", "arguments": {"question": 123}},
        }
    )

    assert "error" in response
    assert response["error"]["code"] == -32602
    assert "string" in response["error"]["message"]


def test_mcp_ask_rejects_malformed_service_result(monkeypatch) -> None:
    class FakeRagService:
        def ask(self, question, top_k, filters, retrieval_mode):
            _ = (question, top_k, filters, retrieval_mode)
            return {
                "answer": "runtime answer",
                "docs": [Document(id="doc:1", content="context")],
                "scores": [],
            }

    class FakeContainer:
        settings_obj = SimpleNamespace(retrieval_mode="sparse")

        def build_rag_service(self):
            return FakeRagService()

    monkeypatch.setattr(mcp_server, "get_cli_container", lambda: FakeContainer())

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 25,
            "method": "tools/call",
            "params": {"name": "rag_ask", "arguments": {"question": "hello"}},
        }
    )

    assert response["result"]["isError"] is True
    assert (
        "RAG service contract violated: 1 docs != 0 scores"
        in response["result"]["content"][0]["text"]
    )


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
    assert {
        doc.external_id for doc in SqlDocumentStorage(in_memory_sqlite).get_all_documents()
    } == {"doc-1"}

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

    assert response["result"]["isError"] is True
    assert "schema_version='4'" in response["result"]["content"][0]["text"]


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


@pytest.mark.parametrize("values", ["demo", [], [" "], ["", "demo"], [1], [None], {"demo": True}])
def test_mcp_invalid_filter_values_are_protocol_argument_errors(values):
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 30,
            "method": "tools/call",
            "params": {
                "name": "rag_ask",
                "arguments": {
                    "question": "hello",
                    "filters": [{"field": "scope", "values": values}],
                },
            },
        }
    )
    assert response["error"]["code"] == -32602


def test_mcp_filter_values_trim_without_splitting_strings():
    filters = mcp_server._parse_filters([{"field": "scope", "values": [" demo "]}])
    assert filters[0].values == ("demo",)


@pytest.mark.parametrize(
    "message",
    [
        None,
        [],
        7,
        {},
        {"jsonrpc": "1.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "id": None, "method": "ping"},
        {"jsonrpc": "2.0", "id": True, "method": "ping"},
        {"jsonrpc": "2.0", "id": [], "method": "ping"},
        {"jsonrpc": "2.0", "id": 1, "method": []},
    ],
)
def test_mcp_invalid_request_envelopes(message):
    assert handle_request(message)["error"]["code"] == -32600


def test_mcp_notifications_have_no_reply_and_ping_is_supported():
    assert handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert (
        handle_request(
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}
        )
        is None
    )
    assert handle_request({"jsonrpc": "2.0", "id": "p", "method": "ping"}) == {
        "jsonrpc": "2.0",
        "id": "p",
        "result": {},
    }


def test_mcp_malformed_call_names_and_arguments_are_recoverable():
    for params in (
        [],
        {"name": []},
        {"name": "missing"},
        {"name": "rag_status", "arguments": {"unknown": 1}},
    ):
        response = handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params}
        )
        assert response["error"]["code"] == -32602


def test_mcp_stdio_recovers_after_bad_json_and_suppresses_notifications(monkeypatch):
    source = io.StringIO(
        '{broken\n{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
        '{"jsonrpc":"2.0","id":2,"method":"ping"}\n'
    )
    output = io.StringIO()
    monkeypatch.setattr(mcp_server.sys, "stdin", source)
    monkeypatch.setattr(mcp_server.sys, "stdout", output)
    mcp_server.main()
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(replies) == 2
    assert replies[0]["error"]["code"] == -32700
    assert replies[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_mcp_stdio_recovers_after_oversized_numeric_id(monkeypatch):
    source = io.StringIO(
        '{"jsonrpc":"2.0","id":'
        + "9" * 5000
        + ',"method":"ping"}\n{"jsonrpc":"2.0","id":2,"method":"ping"}\n'
    )
    output = io.StringIO()
    monkeypatch.setattr(mcp_server.sys, "stdin", source)
    monkeypatch.setattr(mcp_server.sys, "stdout", output)
    previous_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(4300)
        mcp_server.main()
    finally:
        sys.set_int_max_str_digits(previous_limit)
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(replies) == 2
    assert replies[0]["id"] is None
    assert replies[0]["error"]["code"] == -32700
    assert replies[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_mcp_broken_output_is_fatal(monkeypatch):
    class BrokenOutput:
        def write(self, value):
            raise BrokenPipeError("closed")

    monkeypatch.setattr(
        mcp_server.sys, "stdin", io.StringIO('{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
    )
    monkeypatch.setattr(mcp_server.sys, "stdout", BrokenOutput())
    with pytest.raises(BrokenPipeError):
        mcp_server.main()


def test_mcp_recovery_failure_is_a_tool_error(monkeypatch):
    from local_rag_backend.core.errors import MutationRecoveryRequiredError

    def fail():
        raise MutationRecoveryRequiredError(
            "journal.jsonl op_id=op-1 state=pending requires recovery"
        )

    monkeypatch.setitem(mcp_server.TOOLS["rag_status"], "handler", fail)
    response = handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "rag_status"}}
    )
    assert "error" not in response
    assert response["result"]["isError"] is True
    assert "op-1" in response["result"]["content"][0]["text"]
