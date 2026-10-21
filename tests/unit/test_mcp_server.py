from __future__ import annotations

import json
import logging
import select
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from mcp import McpError, types

from local_rag_backend import mcp_server
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.errors import MutationRecoveryRequiredError
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.mcp_transport import MAX_RESULT_BYTES, configure_sdk_logging
from local_rag_backend.settings import get_settings

settings = get_settings()


async def _call(
    server: Any, name: str, arguments: dict[str, Any] | None = None
) -> types.CallToolResult:
    response = await server.request_handlers[types.CallToolRequest](
        types.CallToolRequest(params=types.CallToolRequestParams(name=name, arguments=arguments))
    )
    assert isinstance(response.root, types.CallToolResult)
    assert isinstance(response.root.content[0], types.TextContent)
    assert json.loads(response.root.content[0].text) == response.root.structuredContent
    return response.root


def _invoke(name: str, arguments: dict[str, Any] | None = None) -> types.CallToolResult:
    async def check() -> types.CallToolResult:
        return await _call(mcp_server.create_server(), name, arguments)

    return anyio.run(check)


def test_sdk_discovery_declares_schemas_and_actual_effects() -> None:
    async def check() -> None:
        server = mcp_server.create_server()
        response = await server.request_handlers[types.ListToolsRequest](types.ListToolsRequest())
        assert isinstance(response.root, types.ListToolsResult)
        assert {tool.name for tool in response.root.tools} == set(mcp_server.TOOLS)
        for tool in response.root.tools:
            assert tool.inputSchema and tool.outputSchema and tool.annotations
            assert tool.annotations.readOnlyHint is False
            assert tool.annotations.destructiveHint == (
                tool.name in {"rag_import_canonical", "rag_rebuild_index", "rag_eval"}
            )

    anyio.run(check)


def test_ask_preserves_native_projection_and_semantic_validation(monkeypatch) -> None:
    calls = []

    class Service:
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

    class Container:
        settings_obj = SimpleNamespace(retrieval_mode="sparse")

        def build_rag_service(self):
            return Service()

    monkeypatch.setattr(mcp_server, "get_cli_container", Container)
    result = _invoke(
        "rag_ask",
        {
            "question": "  indexed?  ",
            "k": 1,
            "filters": [{"field": "metadata.scope", "values": [" demo "]}],
        },
    )
    assert not result.isError
    assert result.structuredContent == {
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
    assert calls[0][0:2] == ("indexed?", 1)
    assert calls[0][2][0].field == "metadata.scope"
    assert calls[0][2][0].values == ("demo",)
    assert calls[0][3] == "sparse"


@pytest.mark.parametrize(
    "arguments",
    [
        {"question": "hello", "k": 0},
        {"question": "hello", "k": True},
        {"question": "hello", "k": "1"},
        {"question": "hello", "k": 1.0},
        {"question": "x" * (mcp_server.MAX_ASK_QUESTION_CHARS + 1)},
        {"question": 1},
        {"question": " "},
        {"question": "hello", "unexpected": True},
        *(
            {"question": "hello", "filters": [{"field": "scope", "values": values}]}
            for values in ["demo", [], [" "], ["", "demo"], [1], [None], {"demo": True}]
        ),
    ],
)
def test_invalid_tool_arguments_are_safe_protocol_errors(arguments, monkeypatch) -> None:
    def fail(**kwargs):
        pytest.fail("invalid arguments reached the application")

    monkeypatch.setitem(
        mcp_server.TOOLS, "rag_ask", replace(mcp_server.TOOLS["rag_ask"], handler=fail)
    )
    with pytest.raises(McpError) as caught:
        _invoke("rag_ask", arguments)
    assert caught.value.error.code == -32602
    assert caught.value.error.message == "Invalid tool arguments"


def test_unknown_tool_is_safe_protocol_error() -> None:
    with pytest.raises(McpError) as caught:
        _invoke("PRIVATE_UNKNOWN_TOOL")
    assert caught.value.error.code == -32602
    assert "PRIVATE" not in caught.value.error.message


def test_import_and_status_use_native_mutation_flow(
    in_memory_sqlite, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    settings.data_dir.mkdir()
    result = _invoke(
        "rag_import_canonical",
        {
            "payload": {
                "scope": "generic:demo",
                "snapshot_id": "snap-1",
                "documents": [
                    {
                        "external_id": "doc-1",
                        "content": "def helper(): return 1",
                        "metadata": {"unit_type": "function"},
                    }
                ],
            }
        },
    )
    assert not result.isError and result.structuredContent
    assert result.structuredContent["inserted"] == 1
    assert result.structuredContent["replace_scope"] is True
    assert {
        doc.external_id for doc in SqlDocumentStorage(in_memory_sqlite).get_all_documents()
    } == {"doc-1"}
    status = _invoke("rag_status")
    assert not status.isError and status.structuredContent
    assert status.structuredContent["runtime"]["backends"]["retrieval"] == "sparse"
    assert status.structuredContent["health"]["documents_count"] == 1
    assert status.structuredContent["health"]["history_count"] == 0


def test_invalid_canonical_envelope_is_expected_failure_before_mutation(monkeypatch) -> None:
    def forbidden():
        pytest.fail("invalid canonical payload reached runtime initialization")

    monkeypatch.setattr(mcp_server, "get_cli_container", forbidden)
    result = _invoke(
        "rag_import_canonical",
        {"payload": {"kind": "code-units", "schema_version": "3", "documents": []}},
    )
    assert result.isError and result.structuredContent
    assert "schema_version='6'" in result.structuredContent["error"]["message"]


def test_rebuild_wrong_mode_and_missing_dataset_are_expected_failures(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        mcp_server,
        "get_cli_runtime_snapshot",
        lambda: SimpleNamespace(
            retrieval_mode="sparse", eval_dataset_path=str(tmp_path / "absent.jsonl")
        ),
    )
    assert _invoke("rag_rebuild_index").isError
    result = _invoke("rag_eval")
    assert result.isError and result.structuredContent
    assert "Dataset not found" in result.structuredContent["error"]["message"]


def test_eval_preserves_filters_and_native_result(in_memory_sqlite, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    settings.data_dir.mkdir()
    dataset = Path(__file__).resolve().parents[2] / "datasets/repogpt_rag_eval_v1.jsonl"
    result = _invoke(
        "rag_eval",
        {
            "dataset_path": str(dataset),
            "retrieval_mode": "sparse",
            "k": 1,
            "filters": [{"field": "source_id", "values": ["eval:repogpt:v1"]}],
        },
    )
    assert not result.isError and result.structuredContent
    payload = result.structuredContent
    assert payload["dataset_id"] == "repogpt_rag_eval_v1"
    assert payload["k"] == 1 and payload["queries"] > 0
    assert set(payload["metrics"]) == {"nDCG@1", "MAP@1", "MRR@1", "P@1", "Recall@1"}
    assert payload["filters"] == [{"field": "source_id", "values": ["eval:repogpt:v1"]}]


def test_recovery_required_and_partial_diagnostics_retain_evidence(monkeypatch) -> None:
    def fail():
        raise MutationRecoveryRequiredError("op_id=op-1 state=pending requires recovery")

    monkeypatch.setitem(
        mcp_server.TOOLS, "rag_status", replace(mcp_server.TOOLS["rag_status"], handler=fail)
    )
    result = _invoke("rag_status")
    assert result.isError and result.structuredContent
    assert "op-1" in result.structuredContent["error"]["message"]
    monkeypatch.setitem(
        mcp_server.TOOLS,
        "rag_status",
        replace(
            mcp_server.TOOLS["rag_status"],
            handler=lambda: {
                "version": "4.0.0",
                "runtime": {},
                "health": {"documents_count": 1, "history_error": "unavailable"},
            },
        ),
    )
    partial = _invoke("rag_status")
    assert partial.isError and partial.structuredContent
    assert partial.structuredContent["health"]["documents_count"] == 1
    assert partial.structuredContent["error"]["code"] == "partial_status"


@pytest.mark.parametrize("failure", ["runtime", "value", "output", "nan"])
def test_application_bugs_and_invalid_outputs_are_sanitized(failure, monkeypatch, caplog) -> None:
    def broken(question, k, filters):
        if failure == "runtime":
            raise RuntimeError("PRIVATE_MESSAGE")
        if failure == "value":
            raise ValueError("PRIVATE_MESSAGE")
        if failure == "output":
            return {"answer": 9, "sources": []}
        return {
            "answer": "PRIVATE_MESSAGE",
            "sources": [
                {
                    "document": {
                        "id": "x",
                        "content": "x",
                        "external_id": None,
                        "source_id": None,
                        "metadata": None,
                    },
                    "score": float("nan"),
                }
            ],
        }

    monkeypatch.setitem(
        mcp_server.TOOLS, "rag_ask", replace(mcp_server.TOOLS["rag_ask"], handler=broken)
    )
    with pytest.raises(McpError) as caught:
        _invoke("rag_ask", {"question": "PRIVATE_QUERY"})
    assert caught.value.error.code == -32603
    assert caught.value.error.message == "Internal tool error"
    assert "PRIVATE" not in caplog.text


def test_malformed_rag_service_contract_is_internal_error(monkeypatch) -> None:
    class Service:
        def ask(self, *args, **kwargs):
            return {
                "answer": "PRIVATE",
                "docs": [Document(id="doc:1", content="PRIVATE")],
                "scores": [],
            }

    class Container:
        settings_obj = SimpleNamespace(retrieval_mode="sparse")

        def build_rag_service(self):
            return Service()

    monkeypatch.setattr(mcp_server, "get_cli_container", Container)
    with pytest.raises(McpError) as caught:
        _invoke("rag_ask", {"question": "query"})
    assert caught.value.error.code == -32603
    assert "PRIVATE" not in caught.value.error.message


def test_limit_bounds_full_result_including_both_text_and_structured_payload(monkeypatch) -> None:
    answer = "a" * (140 * 1024)
    assert len(json.dumps({"answer": answer, "sources": []})) < MAX_RESULT_BYTES
    monkeypatch.setitem(
        mcp_server.TOOLS,
        "rag_ask",
        replace(
            mcp_server.TOOLS["rag_ask"], handler=lambda **kwargs: {"answer": answer, "sources": []}
        ),
    )
    result = _invoke("rag_ask", {"question": "query"})
    assert result.isError and result.structuredContent
    assert result.structuredContent["error"]["code"] == "response_too_large"
    assert len(result.model_dump_json().encode()) <= MAX_RESULT_BYTES
    assert "may have completed" in result.structuredContent["error"]["message"]


def test_oversized_mutation_result_keeps_bounded_receipt(monkeypatch) -> None:
    payload = {
        "scope": "generic:demo",
        "snapshot_id": "snap-1",
        "replace_scope": True,
        "inserted": 1,
        "updated": 2,
        "unchanged": 3,
        "deleted_sql": 4,
        "deleted_index": 4,
        "deleted_external_ids": ["sensitive-external-id-" + str(index) for index in range(10000)],
    }
    monkeypatch.setitem(
        mcp_server.TOOLS,
        "rag_import_canonical",
        replace(
            mcp_server.TOOLS["rag_import_canonical"],
            handler=lambda **kwargs: payload,
        ),
    )
    result = _invoke("rag_import_canonical", {"payload": {}})
    assert result.isError and result.structuredContent
    assert result.structuredContent["inline_details_omitted"] is True
    assert result.structuredContent["result_summary"] == {
        key: value for key, value in payload.items() if key != "deleted_external_ids"
    } | {"deleted_external_ids_count": 10000}
    assert len(result.model_dump_json().encode()) < MAX_RESULT_BYTES


def test_sdk_root_and_namespace_logs_are_sanitized(caplog) -> None:
    configure_sdk_logging()
    for name in ("root", "mcp.shared.session"):
        record = logging.LogRecord(
            name,
            logging.WARNING,
            "/package/mcp/shared/session.py",
            1,
            "PRIVATE %s",
            ("PRIVATE",),
            None,
        )
        record.exc_text = "PRIVATE_TRACEBACK"
        record.stack_info = "PRIVATE_STACK"
        logging.getLogger(name).handle(record)
    assert "MCP SDK diagnostic" in caplog.text and "PRIVATE" not in caplog.text


def test_cancelled_work_retains_slot_and_ping_remains_responsive(monkeypatch) -> None:
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked(question, **kwargs):
        calls.append(question)
        entered.set()
        assert release.wait(10)
        return {"answer": "done", "sources": []}

    monkeypatch.setitem(
        mcp_server.TOOLS, "rag_ask", replace(mcp_server.TOOLS["rag_ask"], handler=blocked)
    )

    async def check() -> None:
        server = mcp_server.create_server()
        scope = anyio.CancelScope()

        async def first() -> None:
            with scope:
                await _call(server, "rag_ask", {"question": "first"})

        async def second() -> None:
            await _call(server, "rag_ask", {"question": "second"})

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(first)
                await anyio.to_thread.run_sync(entered.wait)
                scope.cancel()
                group.start_soon(second)
                await anyio.sleep(0.02)
                assert calls == ["first"]
                result = await server.request_handlers[types.PingRequest](types.PingRequest())
                assert isinstance(result.root, types.EmptyResult)
                release.set()
        finally:
            release.set()
        assert calls == ["first", "second"]

    anyio.run(check)


def test_stdio_handshake_ping_notifications_and_recoverable_failures(tmp_path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "local_rag_backend.mcp_server"],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdin and process.stdout

    def send(message):
        process.stdin.write((json.dumps({"jsonrpc": "2.0", **message}) + "\n").encode())
        process.stdin.flush()

    def receive(timeout=20):
        while True:
            ready, _, _ = select.select([process.stdout], [], [], timeout)
            assert ready, "stdio response timed out"
            message = json.loads(process.stdout.readline())
            if "id" in message:
                return message
            assert message["method"] == "notifications/message"
            assert "PRIVATE" not in json.dumps(message)

    try:
        send(
            {
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            }
        )
        assert receive(timeout=60)["result"]["protocolVersion"] == "2025-11-25"
        send({"method": "notifications/initialized"})
        send(
            {
                "method": "tools/call",
                "params": {
                    "name": "rag_import_canonical",
                    "arguments": {"payload": {"PRIVATE_MESSAGE": "PRIVATE_PARTICIPANT"}},
                },
            }
        )
        process.stdin.write(b"{broken PRIVATE_MESSAGE\n")
        process.stdin.flush()
        send({"id": 2, "method": "ping"})
        assert receive() == {"jsonrpc": "2.0", "id": 2, "result": {}}
        assert not (tmp_path / "data").exists()
        send(
            {
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "rag_import_canonical",
                    "arguments": {
                        "payload": {"kind": "unknown", "schema_version": "9", "documents": []}
                    },
                },
            }
        )
        failed = receive()
        assert failed["result"]["isError"] is True
        assert failed["result"]["structuredContent"] == json.loads(
            failed["result"]["content"][0]["text"]
        )
        send({"id": 4, "method": "tools/list"})
        assert len(receive()["result"]["tools"]) == 5
    finally:
        process.terminate()
        _, stderr = process.communicate(timeout=10)
    assert b"PRIVATE" not in stderr


def test_base_mcp_domain_import_does_not_import_sdk() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import local_rag_backend.mcp_server; assert 'mcp' not in sys.modules; assert 'local_rag_backend.mcp_transport' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
