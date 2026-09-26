"""MCP server for rag-prototype operational workflows."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from local_rag_backend import __version__
from local_rag_backend.cli_commands.runtime import (
    build_dense_embedder,
    ensure_sqlite_schema_for_cli,
    get_cli_container,
    get_cli_runtime_snapshot,
    run_cli_mutation,
)
from local_rag_backend.composition.factory import reset_app_context
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalFilter, normalize_filter_field
from local_rag_backend.core.services.canonical_import_transport import (
    build_canonical_import_request_input_from_raw,
)
from local_rag_backend.core.services.evaluation import (
    build_eval_retrieval_config,
    eval_result_to_json,
    load_eval_dataset,
)
from local_rag_backend.core.services.retrieval_filters import validate_filter_values
from local_rag_backend.core.use_cases.docs_import_canonical import (
    execute_import_canonical_sync,
)
from local_rag_backend.core.use_cases.evaluation import run_retrieval_eval

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("rag_prototype_mcp")

MAX_ASK_QUESTION_CHARS = 4096


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _FilterArguments(_Arguments):
    field: str = Field(min_length=1, max_length=256)
    values: list[str] = Field(min_length=1)

    @field_validator("field")
    @classmethod
    def _supported_field(cls, value: str) -> str:
        return normalize_filter_field(value)

    @field_validator("values", mode="before")
    @classmethod
    def _nonblank_values(cls, value: object) -> list[str]:
        return validate_filter_values(value)


class _AskArguments(_Arguments):
    question: str = Field(min_length=1, max_length=MAX_ASK_QUESTION_CHARS)
    k: int = Field(default=3, ge=1, le=10)
    filters: list[_FilterArguments] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class _ImportArguments(_Arguments):
    payload: dict[str, Any]
    replace_scope_override: bool | None = None


class _EvalArguments(_Arguments):
    dataset_path: str | None = None
    retrieval_mode: Literal["sparse", "dense", "dual", "hybrid"] | None = None
    k: int = Field(default=3, ge=1)
    filters: list[_FilterArguments] = Field(default_factory=list)


def _parse_filters(raw_filters: object | None) -> tuple[RetrievalFilter, ...]:
    if raw_filters is None:
        return ()
    if not isinstance(raw_filters, list):
        raise ValueError("filters must be an array of {field, values} objects")
    parsed = [_FilterArguments.model_validate(item) for item in raw_filters]
    return tuple(RetrievalFilter(field=item.field, values=tuple(item.values)) for item in parsed)


def tool_status() -> dict[str, object]:
    container = get_cli_container()
    ensure_sqlite_schema_for_cli()
    readiness_bundle = container.build_health_readiness_bundle()
    diagnostics = readiness_bundle.diagnostics
    runtime = get_cli_runtime_snapshot()

    response: dict[str, object] = {
        "version": __version__,
        "runtime": runtime.to_dict(),
        "health": {},
    }
    health = cast("dict[str, object]", response["health"])

    try:
        health["documents_count"] = diagnostics.get_documents_count()
    except Exception as exc:
        health["documents_error"] = str(exc)

    try:
        health["history_count"] = diagnostics.get_history_count()
    except Exception as exc:
        health["history_error"] = str(exc)

    index: dict[str, object] | None = None
    if runtime.retrieval_mode in {"dense", "dual", "hybrid"}:
        try:
            index = diagnostics.get_retrieval_index_stats(
                index_path=runtime.index_path,
                id_map_path=runtime.id_map_path,
                vector_backend=runtime.vector_backend,
                dim=None,
                expected_manifest=readiness_bundle.expected_manifest,
            )
        except Exception as exc:
            response["index_error"] = str(exc)
    if index is not None:
        response["index"] = index

    return response


def _document_to_json(document: Document) -> dict[str, object]:
    return {
        "id": str(document.id),
        "content": str(document.content),
        "external_id": (
            str(document.external_id)
            if getattr(document, "external_id", None) is not None
            else None
        ),
        "source_id": (
            str(document.source_id) if getattr(document, "source_id", None) is not None else None
        ),
        "metadata": (
            dict(document.metadata or {})
            if getattr(document, "metadata", None) is not None
            else None
        ),
    }


def tool_ask(
    question: str,
    k: int = 3,
    filters: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    if len(question) > MAX_ASK_QUESTION_CHARS:
        raise ValueError(f"question must not exceed {MAX_ASK_QUESTION_CHARS} characters")
    rendered_question = question.strip()
    if not rendered_question:
        raise ValueError("question must not be blank")
    if type(k) is not int:
        raise ValueError("k must be an integer")
    if k < 1 or k > 10:
        raise ValueError("k must be between 1 and 10")

    parsed_filters = _parse_filters(filters)
    container = get_cli_container()
    service = container.build_rag_service()
    result = service.ask(
        rendered_question,
        top_k=k,
        filters=parsed_filters,
        retrieval_mode=str(container.settings_obj.retrieval_mode),
    )
    docs = list(result["docs"])
    scores = list(result["scores"])
    if len(docs) != len(scores):
        raise RuntimeError(
            f"RAG service contract violated: {len(docs)} docs != {len(scores)} scores."
        )
    return {
        "answer": str(result["answer"]),
        "sources": [
            {"document": _document_to_json(doc), "score": float(score)}
            for doc, score in zip(docs, scores, strict=True)
        ],
    }


def tool_import_canonical(
    payload: dict[str, Any],
    replace_scope_override: bool | None = None,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object.")
    try:
        request = build_canonical_import_request_input_from_raw(
            payload,
            source="mcp:rag_import_canonical",
            replace_scope_override=replace_scope_override,
        )
    except (ValidationError, ValueError) as exc:
        raise ValueError(str(exc)) from exc

    container = get_cli_container()
    mutation_bundle = container.build_docs_mutation_bundle()

    def _run_sync() -> dict[str, object]:
        summary = execute_import_canonical_sync(
            request=request,
            settings_obj=container.settings_obj,
            ports=mutation_bundle.ports,
        )
        return {
            "scope": summary.scope,
            "snapshot_id": summary.snapshot_id,
            "replace_scope": summary.replace_scope,
            "inserted": summary.inserted,
            "updated": summary.updated,
            "unchanged": summary.unchanged,
            "deleted_sql": summary.deleted_sql,
            "deleted_index": summary.deleted_index,
            "deleted_external_ids": list(summary.deleted_external_ids or []),
        }

    result: dict[str, object] = run_cli_mutation(_run_sync, use_lock=True)
    return result


def tool_rebuild_index() -> dict[str, object]:
    runtime = get_cli_runtime_snapshot()
    if runtime.retrieval_mode not in {"dense", "dual", "hybrid"}:
        raise RuntimeError("rag_rebuild_index requires retrieval_mode=dense|dual|hybrid.")

    from local_rag_backend.core.use_cases import index as index_service

    container = get_cli_container()
    ports = container.index_mutation_ports(build_embedder=build_dense_embedder)

    def _rebuild_sync() -> int:
        return int(
            index_service.rebuild_index_sync(settings_obj=container.settings_obj, ports=ports)
        )

    rebuilt = run_cli_mutation(_rebuild_sync)
    return {"vectors": rebuilt, "retrieval_mode": runtime.retrieval_mode}


def tool_eval(
    dataset_path: str | None = None,
    retrieval_mode: str | None = None,
    k: int | None = None,
    filters: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    runtime = get_cli_runtime_snapshot()
    dataset = load_eval_dataset(dataset_path or runtime.eval_dataset_path)
    parsed_filters = _parse_filters(filters)
    config = build_eval_retrieval_config(
        retrieval_mode=retrieval_mode,
        k=(3 if k is None else k),
    )

    container = get_cli_container()
    eval_bundle = container.build_eval_execution_bundle()
    result = run_retrieval_eval(
        dataset=dataset,
        eval_storage_port=eval_bundle.eval_storage_port,
        eval_retriever_factory_port=eval_bundle.eval_retriever_factory_port,
        retrieval_mode=config.retrieval_mode,
        k=config.k,
        reranker_enabled=False,
        reranker_candidate_k=eval_bundle.reranker_candidate_k,
        reranker_strategy=eval_bundle.reranker_strategy,
        filters=parsed_filters,
    )
    response = eval_result_to_json(result)
    response["filters"] = [
        {"field": item.field, "values": list(item.values)} for item in parsed_filters
    ]
    return response


TOOLS: dict[str, dict[str, Any]] = {
    "rag_status": {
        "description": "Return structured runtime status, counts, and index diagnostics.",
        "arguments": _Arguments,
        "handler": tool_status,
    },
    "rag_ask": {
        "description": "Ask a question using the configured RAG runtime.",
        "arguments": _AskArguments,
        "handler": tool_ask,
    },
    "rag_import_canonical": {
        "description": "Import canonical documents via the native scope/snapshot sync flow.",
        "arguments": _ImportArguments,
        "handler": tool_import_canonical,
    },
    "rag_rebuild_index": {
        "description": "Rebuild the dense/dual/hybrid retrieval index.",
        "arguments": _Arguments,
        "handler": tool_rebuild_index,
    },
    "rag_eval": {
        "description": "Run offline retrieval evaluation with optional retrieval filters.",
        "arguments": _EvalArguments,
        "handler": tool_eval,
    },
}

PROTOCOL_VERSION = "2024-11-05"


def _protocol_error(req_id: object, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_request(request: object) -> dict[str, Any] | None:
    """Handle one MCP message; notifications deliberately produce no response."""
    if not isinstance(request, dict):
        return _protocol_error(None, -32600, "Request must be a JSON object")
    req_id = request.get("id")
    method = request.get("method")
    if (
        request.get("jsonrpc") != "2.0"
        or not isinstance(method, str)
        or not method
        or ("id" in request and type(req_id) not in (int, str))
    ):
        return _protocol_error(None, -32600, "Invalid JSON-RPC request")
    if "id" not in request:
        return None
    params = request.get("params", {})
    if not isinstance(params, dict):
        return _protocol_error(req_id, -32602, "params must be an object")

    if method == "initialize":
        if (
            not isinstance(params.get("protocolVersion"), str)
            or not isinstance(params.get("capabilities"), dict)
            or not isinstance(params.get("clientInfo"), dict)
            or not isinstance(params["clientInfo"].get("name"), str)
            or not isinstance(params["clientInfo"].get("version"), str)
        ):
            return _protocol_error(req_id, -32602, "Invalid initialize parameters")
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": "rag-prototype", "version": __version__},
            "capabilities": {"tools": {}},
        }
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": name,
                        "description": spec["description"],
                        "inputSchema": spec["arguments"].model_json_schema(),
                    }
                    for name, spec in TOOLS.items()
                ]
            },
        }

    if method != "tools/call":
        return _protocol_error(req_id, -32601, f"Unknown method: {method}")
    tool_name = params.get("name")
    if not isinstance(tool_name, str) or tool_name not in TOOLS:
        return _protocol_error(req_id, -32602, f"Unknown tool: {tool_name}")
    spec = TOOLS[tool_name]
    try:
        arguments = spec["arguments"].model_validate(params.get("arguments", {}))
    except ValidationError as exc:
        return _protocol_error(req_id, -32602, str(exc))
    try:
        result = spec["handler"](**arguments.model_dump())
        tool_result: dict[str, Any] = {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
        }
    except Exception as exc:
        logger.error("Tool error in %s: %s", tool_name, exc)
        tool_result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
    return {"jsonrpc": "2.0", "id": req_id, "result": tool_result}


def main() -> None:
    logger.info("rag-prototype MCP Server starting...")
    response: dict[str, Any] | None
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                response = _protocol_error(None, -32700, f"Parse error: {exc}")
            else:
                response = handle_request(request)
            if response is not None:
                # Broken streams are fatal. Continuing cannot deliver a response and can spin.
                print(json.dumps(response), flush=True)
    finally:
        reset_app_context()


if __name__ == "__main__":
    main()
