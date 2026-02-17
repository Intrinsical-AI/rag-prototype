"""Document-related CLI commands."""

from local_rag_backend.cli_commands.docs_bootstrap import bootstrap_cmd
from local_rag_backend.cli_commands.docs_delete import delete_docs_cmd, delete_external_ids_cmd
from local_rag_backend.cli_commands.docs_ingest import ingest_cmd
from local_rag_backend.cli_commands.docs_upsert import upsert_docs_cmd

__all__ = [
    "bootstrap_cmd",
    "delete_docs_cmd",
    "delete_external_ids_cmd",
    "ingest_cmd",
    "upsert_docs_cmd",
]

