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
from local_rag_backend.core.domain.entities import Document as DomainDocument
from local_rag_backend.core.domain.retrieval import (
    RetrievalFilter,
    RetrievalRequest,
    RetrievalResult,
    retrieval_result_from_pairs,
)
from local_rag_backend.core.services.canonical_import_transport import (
    build_canonical_import_request_input_from_raw,
)
from local_rag_backend.core.services.evaluation import (
    EvalResult,
    load_eval_dataset,
    run_retrieval_eval as run_retrieval_eval_core,
)
from local_rag_backend.core.services.types import EvalRetrievalConfig, EvalRetrievalMode
from local_rag_backend.core.use_cases.docs_import_canonical import (
    execute_import_canonical_sync,
)
from local_rag_backend.core.use_cases.evaluation import prepare_eval_workspace

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("rag_prototype_mcp")


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


def _coerce_retrieval_result(
    raw_result: object,
    *,
    request: RetrievalRequest,
) -> RetrievalResult:
    if isinstance(raw_result, RetrievalResult):
        return raw_result
    if (
        isinstance(raw_result, tuple)
        and len(raw_result) == 2
        and isinstance(raw_result[0], (list, tuple))
        and isinstance(raw_result[1], (list, tuple))
    ):
        docs, scores = raw_result
        return retrieval_result_from_pairs(
            docs=cast("Sequence[DomainDocument]", docs),
            scores=cast("Sequence[float]", scores),
            mode_used=request.mode,
            backend_used="eval",
        )
    raise RuntimeError(f"Unsupported eval retriever response type: {type(raw_result)!r}")


def _eval_result_to_dict(result: EvalResult) -> dict[str, object]:
    return {
        "dataset_id": result.dataset_id,
        "retrieval_mode": result.retrieval_mode,
        "reranker_enabled": result.reranker_enabled,
        "k": result.k,
        "queries": result.queries,
        "ndcg_at_k": result.ndcg_at_k,
        "map_at_k": result.map_at_k,
        "mrr_at_k": result.mrr_at_k,
        "precision_at_k": result.precision_at_k,
        "recall_at_k": result.recall_at_k,
    }


def _resolve_eval_mode(raw_mode: object | None) -> EvalRetrievalMode:
    mode = str(raw_mode or "sparse").strip().lower()
    if mode not in {"sparse", "dense", "dual", "hybrid"}:
        raise ValueError("retrieval_mode must be one of sparse, dense, dual, hybrid.")
    return cast("EvalRetrievalMode", mode)


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

    return run_cli_mutation(_run_sync, use_lock=True)


def tool_rebuild_index() -> dict[str, object]:
    runtime = get_cli_runtime_snapshot()
    if runtime.retrieval_mode not in {"dense", "dual", "hybrid"}:
        raise RuntimeError("rag_rebuild_index requires retrieval_mode=dense|dual|hybrid.")

    from local_rag_backend.core.use_cases import index as index_service

    container = get_cli_container()
    ports = container.index_mutation_ports(build_embedder=build_dense_embedder)

    def _rebuild_sync() -> int:
        return index_service.rebuild_index_sync(settings_obj=container.settings_obj, ports=ports)

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
    mode = _resolve_eval_mode(retrieval_mode)
    top_k = int(k or 3)
    if top_k <= 0:
        raise ValueError("k must be positive.")

    container = get_cli_container()
    eval_bundle = container.build_eval_execution_bundle()
    workspace = prepare_eval_workspace(
        dataset=dataset,
        eval_storage_port=eval_bundle.eval_storage_port,
        eval_retriever_factory_port=eval_bundle.eval_retriever_factory_port,
    )
    config = EvalRetrievalConfig(retrieval_mode=mode, k=top_k, reranker_enabled=False)
    prepared = workspace.prepared_retriever_workspace
    if prepared is not None and hasattr(prepared, "build_retriever"):
        retriever = prepared.build_retriever(
            config=config,
            reranker_candidate_k=eval_bundle.reranker_candidate_k,
            reranker_strategy=eval_bundle.reranker_strategy,
        )
    else:
        retriever = workspace.eval_retriever_factory_port.build_retriever(
            storage=workspace.eval_storage_port,
            config=config,
            reranker_candidate_k=eval_bundle.reranker_candidate_k,
            reranker_strategy=eval_bundle.reranker_strategy,
        )

    def _retrieve_external_ids(query: str, requested_top_k: int) -> list[str]:
        request = RetrievalRequest(
            query=query,
            top_k=requested_top_k,
            mode=mode,
            filters=parsed_filters,
        )
        retrieval = _coerce_retrieval_result(retriever.retrieve(request), request=request)
        return [
            str(item.document.external_id)
            for item in retrieval.items
            if getattr(item.document, "external_id", None) is not None
        ]

    result = run_retrieval_eval_core(
        dataset=dataset,
        retrieve_external_ids=_retrieve_external_ids,
        retrieval_mode=mode,
        k=top_k,
        reranker_enabled=False,
        max_queries=None,
        run_out=None,
    )
    response = _eval_result_to_dict(result)
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
