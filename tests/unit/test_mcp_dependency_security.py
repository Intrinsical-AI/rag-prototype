"""Exercise SDK output-schema validation without a server, external I/O or app startup."""

from __future__ import annotations

import io
import json
import socket
import urllib.request
from typing import Any, NoReturn
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp import ClientSession, types


@pytest.mark.parametrize(
    "reference",
    [
        "https://schema.invalid/owned-regression.json",
        "file:///owned-regression.json",
        None,
    ],
    ids=["https-ref-refused", "file-ref-refused", "local-ref-valid"],
)
def test_sdk_tool_output_schema_refuses_external_retrieval(
    monkeypatch: pytest.MonkeyPatch, reference: str | None
) -> None:
    retrievals: list[str] = []

    def controlled_retrieve(
        request: urllib.request.Request | str, *args: Any, **kwargs: Any
    ) -> io.BytesIO:
        # The old SDK may request a resource. Return a harmless local schema so
        # the regression fails on unsafe retrieval without ever performing I/O.
        retrievals.append(
            request.full_url if isinstance(request, urllib.request.Request) else request
        )
        return io.BytesIO(json.dumps({"type": "object"}).encode())

    def no_network(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("External I/O is forbidden in the schema-reference regression")

    async def check() -> None:
        # Create the event loop before guarding the exercised SDK path.
        monkeypatch.setattr(urllib.request, "urlopen", controlled_retrieve)
        monkeypatch.setattr(socket, "getaddrinfo", no_network)
        monkeypatch.setattr(socket, "create_connection", no_network)
        monkeypatch.setattr(socket.socket, "connect", no_network)
        monkeypatch.setattr(socket.socket, "connect_ex", no_network)
        read_send, read_receive = anyio.create_memory_object_stream[Any](1)
        write_send, write_receive = anyio.create_memory_object_stream[Any](1)
        async with read_send, read_receive, write_send, write_receive:
            session = ClientSession(read_receive, write_send)
            schema = (
                {"$ref": reference}
                if reference is not None
                else {
                    "$defs": {"payload": {"type": "object"}},
                    "$ref": "#/$defs/payload",
                }
            )
            result = types.CallToolResult(content=[], structuredContent={"answer": "owned"})
            requests = AsyncMock(
                side_effect=[
                    types.ListToolsResult(
                        tools=[
                            types.Tool(
                                name="owned",
                                inputSchema={"type": "object"},
                                outputSchema=schema,
                            )
                        ]
                    ),
                    result,
                ]
            )
            monkeypatch.setattr(session, "send_request", requests)
            await session.list_tools()
            if reference is not None:
                with pytest.raises(RuntimeError, match="Invalid schema for tool owned"):
                    await session.call_tool("owned", {})
            else:
                assert await session.call_tool("owned", {}) == result
            assert requests.await_count == 2
            assert [call.args[0].root.method for call in requests.await_args_list] == [
                "tools/list",
                "tools/call",
            ]
            assert retrievals == []

    anyio.run(check)
