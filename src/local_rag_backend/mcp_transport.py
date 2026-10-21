"""Official SDK lifecycle with explicit validation, safe failures and bounded work."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

import anyio
from jsonschema import Draft202012Validator
from mcp import McpError, types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import ValidationError

from local_rag_backend import __version__

if TYPE_CHECKING:
    from local_rag_backend.mcp_server import Tool

MAX_RESULT_BYTES = 256 * 1024
logger = logging.getLogger(__name__)
_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["error"],
    "properties": {
        "error": {
            "type": "object",
            "required": ["code", "message"],
            "properties": {"code": {"type": "string"}, "message": {"type": "string"}},
            "additionalProperties": False,
        }
    },
}


class SDKDiagnosticFilter(logging.Filter):
    """SDK namespace/root diagnostics can embed malformed private request bodies."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("mcp") or "/mcp/" in record.pathname.replace("\\", "/"):
            record.msg = "MCP SDK diagnostic"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


def configure_sdk_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    for handler in (*root.handlers, *logging.getLogger("mcp").handlers):
        if not any(isinstance(item, SDKDiagnosticFilter) for item in handler.filters):
            handler.addFilter(SDKDiagnosticFilter())
    logging.getLogger("mcp").setLevel(logging.WARNING)


def _result(payload: dict[str, object], *, failed: bool) -> types.CallToolResult:
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=encoded)],
        structuredContent=payload,
        isError=failed,
    )


def _is_partial_status(name: str, payload: dict[str, object]) -> bool:
    health = payload.get("health")
    return name == "rag_status" and (
        "index_error" in payload
        or (isinstance(health, dict) and any(field.endswith("_error") for field in health))
    )


def _overflow_receipt(
    name: str, payload: dict[str, object], *, operation_returned: bool
) -> dict[str, object]:
    """Retain bounded mutation identifiers/counts while explicitly omitting inline details."""
    summary: dict[str, object] = {}
    if operation_returned:
        for field in (
            "scope",
            "snapshot_id",
            "replace_scope",
            "inserted",
            "updated",
            "unchanged",
            "deleted_sql",
            "deleted_index",
            "vectors",
            "retrieval_mode",
            "dataset_id",
            "k",
            "queries",
        ):
            value = payload.get(field)
            if type(value) in {int, bool} or (isinstance(value, str) and len(value) <= 1024):
                summary[field] = value
        deleted = payload.get("deleted_external_ids")
        if isinstance(deleted, list):
            summary["deleted_external_ids_count"] = len(deleted)
        sources = payload.get("sources")
        if isinstance(sources, list):
            summary["sources_count"] = len(sources)
    return {
        "error": {
            "code": "response_too_large",
            "message": "Tool response exceeds 256 KiB. The operation may have completed; inspect persisted state before retrying.",
        },
        "operation": {"tool": name, "returned": operation_returned},
        "result_summary": summary,
        "inline_details_omitted": True,
    }


def create_sdk_server(
    tools: Mapping[str, Tool], *, expected_errors: tuple[type[Exception], ...]
) -> Server[Any, Any]:
    """One worker slot stays held until its thread finishes, including cancellation."""
    server: Server[Any, Any] = Server("rag-prototype", version=__version__)
    limiter = anyio.CapacityLimiter(1)
    input_schemas = {name: spec.arguments.model_json_schema() for name, spec in tools.items()}
    schemas = {
        name: {"type": "object", "anyOf": [spec.success_schema, _ERROR_SCHEMA]}
        for name, spec in tools.items()
    }
    success = {name: Draft202012Validator(spec.success_schema) for name, spec in tools.items()}
    outputs = {name: Draft202012Validator(schema) for name, schema in schemas.items()}
    for schema in (*input_schemas.values(), *schemas.values()):
        Draft202012Validator.check_schema(schema)

    async def list_tools(_request: types.ListToolsRequest) -> types.ServerResult:
        return types.ServerResult(
            types.ListToolsResult(
                tools=[
                    types.Tool(
                        name=name,
                        description=spec.description,
                        inputSchema=input_schemas[name],
                        outputSchema=schemas[name],
                        annotations=types.ToolAnnotations(
                            readOnlyHint=False,
                            destructiveHint=spec.destructive,
                            idempotentHint=False,
                            openWorldHint=spec.open_world,
                        ),
                    )
                    for name, spec in tools.items()
                ]
            )
        )

    async def call(request: types.CallToolRequest) -> types.ServerResult:
        name = request.params.name
        if name not in tools:
            raise McpError(types.ErrorData(code=-32602, message="Unknown tool"))
        spec = tools[name]
        try:
            arguments = spec.arguments.model_validate(request.params.arguments or {})
        except ValidationError:
            raise McpError(types.ErrorData(code=-32602, message="Invalid tool arguments")) from None
        try:
            failed = False
            operation_returned = False
            try:
                payload = await anyio.to_thread.run_sync(
                    partial(spec.handler, **arguments.model_dump()),
                    abandon_on_cancel=False,
                    limiter=limiter,
                )
                operation_returned = True
                success[name].validate(payload)
                if _is_partial_status(name, payload):
                    failed = True
                    payload = {
                        **payload,
                        "error": {
                            "code": "partial_status",
                            "message": "Some runtime diagnostics failed; inspect retained status.",
                        },
                    }
            except expected_errors as exc:
                payload = {"error": {"code": type(exc).__name__, "message": str(exc)}}
                failed = True
            outputs[name].validate(payload)
            result = _result(payload, failed=failed)
            # Text and structured data are both serialized: bound their combined wire size.
            if (
                len(result.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8"))
                > MAX_RESULT_BYTES
            ):
                payload = _overflow_receipt(name, payload, operation_returned=operation_returned)
                outputs[name].validate(payload)
                result = _result(payload, failed=True)
            return types.ServerResult(result)
        except Exception as exc:
            # Final adapter boundary deliberately sanitizes application bugs and invalid output.
            logger.error("MCP internal failure (%s)", type(exc).__name__)
            raise McpError(types.ErrorData(code=-32603, message="Internal tool error")) from None

    server.request_handlers[types.ListToolsRequest] = list_tools
    server.request_handlers[types.CallToolRequest] = call
    return server


async def _serve(server: Server[Any, Any]) -> None:
    configure_sdk_logging()
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def run_stdio(server: Server[Any, Any]) -> None:
    anyio.run(_serve, server)
