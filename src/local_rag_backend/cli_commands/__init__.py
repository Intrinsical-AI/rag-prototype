"""Domain-organized CLI command modules."""

from local_rag_backend.cli_commands.docs import (
    bootstrap_cmd,
    ingest_cmd,
    mutate_docs_cmd,
)
from local_rag_backend.cli_commands.eval import eval_cmd
from local_rag_backend.cli_commands.index import build_index_cmd, rebuild_index_cmd, status_cmd
from local_rag_backend.cli_commands.server import server_cmd

__all__ = [
    "bootstrap_cmd",
    "build_index_cmd",
    "eval_cmd",
    "ingest_cmd",
    "mutate_docs_cmd",
    "rebuild_index_cmd",
    "server_cmd",
    "status_cmd",
]
