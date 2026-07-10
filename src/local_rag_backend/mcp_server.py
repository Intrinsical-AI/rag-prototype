"""MCP server for rag-prototype operational workflows."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import ValidationError

from local_rag_backend import __version__
from local_rag_backend.cli_commands.runtime import (
    build_dense_embedder,
    ensure_sqlite_schema_for_cli,
    get_cli_container,
    get_cli_runtime_snapshot,
    run_cli_mutation,
)
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalFilter
from local_rag_backend.core.services.canonical_import_transport import (
    build_canonical_import_request_input_from_raw,
)
from local_rag_backend.core.services.evaluation import (
    build_eval_retrieval_config,
    eval_result_to_json,
    load_eval_dataset,
)
from local_rag_backend.core.use_cases.docs_import_canonical import (
    execute_import_canonical_sync,
)
from local_rag_backend.core.use_cases.evaluation import run_retrieval_eval

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("rag_prototype_mcp")

MAX_ASK_QUESTION_CHARS = 4096


def _parse_filters(raw_filters: object | None) -> tuple[RetrievalFilter, ...]:
    if raw_filters is None:
        return ()
    if not isinstance(raw_filters, Sequence) or isinstance(raw_filters, str | bytes):
        raise ValueError("filters must be a list of {field, values} objects.")

    parsed: list[RetrievalFilter] = []
    for idx, item in enumerate(raw_filters):
        if not isinstance(item, Mapping):
            raise ValueError(f"filters[{idx}] must be an object.")
        parsed.append(
            RetrievalFilter(
                field=str(item.get("field") or "").strip(),
                values=tuple(
                    str(value).strip()
                    for value in cast("Sequence[object]", item.get("values") or ())
                    if str(value).strip()
                ),
            )
        )
    return tuple(parsed)


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
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "handler": tool_status,
    },
    "rag_ask": {
        "description": "Ask a question using the configured RAG runtime.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_ASK_QUESTION_CHARS,
                },
                "k": {"type": "integer", "minimum": 1, "maximum": 10},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "values": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["field", "values"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        "handler": tool_ask,
    },
    "rag_import_canonical": {
        "description": "Import canonical documents via the native scope/snapshot sync flow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "payload": {"type": "object", "description": "Canonical import payload."},
                "replace_scope_override": {
                    "type": "boolean",
                    "description": "Optional override for replace_scope.",
                },
            },
            "required": ["payload"],
            "additionalProperties": False,
        },
        "handler": tool_import_canonical,
    },
    "rag_rebuild_index": {
        "description": "Rebuild the dense/dual/hybrid retrieval index.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "handler": tool_rebuild_index,
    },
    "rag_eval": {
        "description": "Run offline retrieval evaluation with optional retrieval filters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_path": {"type": "string"},
                "retrieval_mode": {"type": "string"},
                "k": {"type": "integer"},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "values": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["field", "values"],
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        "handler": tool_eval,
    },
}


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "rag-prototype", "version": __version__},
                "capabilities": {"tools": {}},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": name,
                        "description": spec["description"],
                        "inputSchema": spec["input_schema"],
                    }
                    for name, spec in TOOLS.items()
                ]
            },
        }

    if method == "tools/call":
        if not isinstance(params, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "params must be an object"},
            }
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        if not isinstance(tool_args, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "tool arguments must be an object"},
            }
        try:
            result = TOOLS[str(tool_name)]["handler"](**dict(tool_args))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            }
        except Exception as exc:
            logger.error("Tool error in %s: %s", tool_name, exc)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main() -> None:
    logger.info("rag-prototype MCP Server starting...")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response), flush=True)
        except Exception as exc:  # pragma: no cover - defensive stdio guard
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()
