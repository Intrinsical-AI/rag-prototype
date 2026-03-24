"""Domain-organized CLI command modules."""

from local_rag_backend.cli_commands.docs import (
    bootstrap_cmd,
    import_canonical_cmd,
    ingest_cmd,
    mutate_docs_cmd,
)
from local_rag_backend.cli_commands.eval import eval_batch_cmd, eval_cmd, eval_compare_cmd
from local_rag_backend.cli_commands.index import rebuild_index_cmd, status_cmd
from local_rag_backend.cli_commands.server import server_cmd

__all__ = [
    "bootstrap_cmd",
    "eval_batch_cmd",
    "eval_cmd",
    "eval_compare_cmd",
    "import_canonical_cmd",
    "ingest_cmd",
    "mutate_docs_cmd",
    "rebuild_index_cmd",
    "server_cmd",
    "status_cmd",
]
