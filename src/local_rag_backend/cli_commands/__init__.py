"""Domain-organized CLI command modules."""

from local_rag_backend.cli_commands.docs import (
    bootstrap_cmd,
    delete_docs_cmd,
    delete_external_ids_cmd,
    ingest_cmd,
    upsert_docs_cmd,
)
from local_rag_backend.cli_commands.eval import eval_cmd
from local_rag_backend.cli_commands.index import build_index_cmd, rebuild_index_cmd, status_cmd
from local_rag_backend.cli_commands.server import server_cmd

__all__ = [
    "bootstrap_cmd",
    "build_index_cmd",
    "delete_docs_cmd",
    "delete_external_ids_cmd",
    "eval_cmd",
    "ingest_cmd",
    "rebuild_index_cmd",
    "server_cmd",
    "status_cmd",
    "upsert_docs_cmd",
]
