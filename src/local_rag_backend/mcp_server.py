"""MCP server for rag-prototype operational workflows."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from local_rag_backend import __version__
from local_rag_backend.cli_commands.runtime import (
    build_dense_embedder,
    get_cli_container,
    get_cli_runtime_snapshot,
    run_cli_mutation,
)
from local_rag_backend.composition.ephemeral import build_ephemeral_service, snapshot_status
from local_rag_backend.composition.factory import reset_app_context
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalFilter, normalize_filter_field
from local_rag_backend.core.errors import RagBaseError
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
from local_rag_backend.infrastructure.persistence.read_snapshot import SnapshotError
from local_rag_backend.settings import get_settings

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server


class ToolOperationError(ValueError):
    """Known input/configuration refusal; application bugs retain internal-error handling."""


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
    k: int = Field(default=3, ge=1, le=100)
    filters: list[_FilterArguments] = Field(default_factory=list)


def _parse_filters(raw_filters: object | None) -> tuple[RetrievalFilter, ...]:
    if raw_filters is None:
        return ()
    if not isinstance(raw_filters, list):
        raise ValueError("filters must be an array of {field, values} objects")
    parsed = [_FilterArguments.model_validate(item) for item in raw_filters]
    return tuple(RetrievalFilter(field=item.field, values=tuple(item.values)) for item in parsed)


def tool_status() -> dict[str, object]:
    return {"version": __version__, **snapshot_status(get_settings())}


def tool_ask_ephemeral(
    question: str, k: int = 3, filters: list[dict[str, Any]] | None = None
) -> dict[str, object]:
    arguments = _AskArguments.model_validate(
        {"question": question, "k": k, "filters": filters or []}
    )
    settings = get_settings()
    try:
        with build_ephemeral_service(settings) as (service, snapshot_id):
            result = service.ask(
                arguments.question,
                top_k=arguments.k,
                filters=_parse_filters(arguments.model_dump()["filters"]),
                retrieval_mode=str(settings.retrieval_mode),
                persist_history=False,
            )
    except (SnapshotError, RagBaseError, OSError) as exc:
        raise ToolOperationError(
            "Ephemeral query refused; verify snapshot and configured provider availability"
        ) from exc
    docs, scores = list(result["docs"]), list(result["scores"])
    if len(docs) != len(scores):
        raise RuntimeError("Ephemeral retrieval result contract violated")
    return {
        "answer": str(result["answer"]),
        "sources": [
            {"document": _document_to_json(doc), "score": float(score)}
            for doc, score in zip(docs, scores, strict=True)
        ],
        "snapshot_id": snapshot_id,
    }


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
        raise ToolOperationError(str(exc)) from exc

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
        raise ToolOperationError("rag_rebuild_index requires retrieval_mode=dense|dual|hybrid.")

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
    if k is not None and (type(k) is not int or not 1 <= k <= 100):
        raise ToolOperationError("k must be an integer between 1 and 100")
    runtime = get_cli_runtime_snapshot()
    try:
        dataset = load_eval_dataset(dataset_path or runtime.eval_dataset_path)
    except (OSError, ValueError) as exc:
        raise ToolOperationError(str(exc)) from exc
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


@dataclass(frozen=True)
class Tool:
    description: str
    arguments: type[BaseModel]
    handler: Callable[..., dict[str, object]]
    success_schema: dict[str, Any]
    destructive: bool = False
    open_world: bool = False
    read_only: bool = False


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


_COUNT = {"type": "integer", "minimum": 0}
_TEXT = {"type": "string"}
_NULLABLE_TEXT = {"type": ["string", "null"]}
_DOCUMENT = _schema(
    {
        "id": _TEXT,
        "content": _TEXT,
        "external_id": _NULLABLE_TEXT,
        "source_id": _NULLABLE_TEXT,
        "metadata": {"type": ["object", "null"]},
    },
    ["id", "content", "external_id", "source_id", "metadata"],
)

TOOLS: dict[str, Tool] = {
    "rag_status": Tool(
        "Inspect live store presence and verified frozen snapshot readiness without writing or querying live storage.",
        _Arguments,
        tool_status,
        _schema(
            {"version": _TEXT, "runtime": {"type": "object"}, "health": {"type": "object"}},
            ["version", "runtime", "health"],
        ),
        read_only=True,
    ),
    "rag_ask_ephemeral": Tool(
        "Query a verified frozen snapshot without history, cache or storage writes. Configured provider egress and cost may occur; calls are never retried automatically.",
        _AskArguments,
        tool_ask_ephemeral,
        _schema(
            {
                "answer": _TEXT,
                "snapshot_id": _TEXT,
                "sources": {
                    "type": "array",
                    "items": _schema(
                        {"document": _DOCUMENT, "score": {"type": "number"}}, ["document", "score"]
                    ),
                },
            },
            ["answer", "sources", "snapshot_id"],
        ),
        read_only=True,
        open_world=True,
    ),
    "rag_ask": Tool(
        "Ask the configured RAG runtime and persist QA history; may call a configured LLM.",
        _AskArguments,
        tool_ask,
        _schema(
            {
                "answer": _TEXT,
                "sources": {
                    "type": "array",
                    "items": _schema(
                        {
                            "document": _DOCUMENT,
                            "score": {"type": "number"},
                        },
                        ["document", "score"],
                    ),
                },
            },
            ["answer", "sources"],
        ),
        open_world=True,
    ),
    "rag_import_canonical": Tool(
        "Import canonical documents through the locked native scope/snapshot mutation flow.",
        _ImportArguments,
        tool_import_canonical,
        _schema(
            {
                "scope": _TEXT,
                "snapshot_id": _TEXT,
                "replace_scope": {"type": "boolean"},
                **dict.fromkeys(
                    ("inserted", "updated", "unchanged", "deleted_sql", "deleted_index"), _COUNT
                ),
                "deleted_index": {"type": ["integer", "null"], "minimum": 0},
                "deleted_external_ids": {"type": "array", "items": _TEXT},
            },
            [
                "scope",
                "snapshot_id",
                "replace_scope",
                "inserted",
                "updated",
                "unchanged",
                "deleted_sql",
                "deleted_index",
                "deleted_external_ids",
            ],
        ),
        destructive=True,
        open_world=True,
    ),
    "rag_rebuild_index": Tool(
        "Rebuild the configured dense/dual/hybrid retrieval index using the native mutation lock.",
        _Arguments,
        tool_rebuild_index,
        _schema(
            {"vectors": _COUNT, "retrieval_mode": {"enum": ["dense", "dual", "hybrid"]}},
            ["vectors", "retrieval_mode"],
        ),
        destructive=True,
        open_world=True,
    ),
    "rag_eval": Tool(
        "Run offline retrieval evaluation in its persistent evaluation workspace.",
        _EvalArguments,
        tool_eval,
        _schema(
            {
                "dataset_id": _TEXT,
                "retrieval_mode": {"enum": ["sparse", "dense", "dual", "hybrid"]},
                "reranker_enabled": {"type": "boolean"},
                "k": {"type": "integer", "minimum": 1},
                "queries": _COUNT,
                "metrics": {"type": "object", "additionalProperties": {"type": "number"}},
                "filters": {"type": "array", "items": _FilterArguments.model_json_schema()},
            },
            [
                "dataset_id",
                "retrieval_mode",
                "reranker_enabled",
                "k",
                "queries",
                "metrics",
                "filters",
            ],
        ),
        destructive=True,
        open_world=True,
    ),
}


def create_server() -> Server[Any, Any]:
    """Load the optional SDK only when an MCP server is requested."""
    from local_rag_backend.mcp_transport import create_sdk_server

    return create_sdk_server(TOOLS, expected_errors=(ToolOperationError, RagBaseError, OSError))


def main() -> None:
    try:
        from local_rag_backend.mcp_transport import run_stdio
    except ModuleNotFoundError as exc:
        if exc.name in {"mcp", "anyio", "jsonschema"}:
            sys.stderr.write("rag-mcp requires the optional mcp extra: uv sync --extra mcp\n")
            raise SystemExit(2) from None
        raise
    try:
        run_stdio(create_server())
    finally:
        reset_app_context()


if __name__ == "__main__":
    main()
