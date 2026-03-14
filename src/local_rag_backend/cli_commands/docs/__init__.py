"""Document-related CLI commands."""

from local_rag_backend.cli_commands.docs.docs_bootstrap import bootstrap_cmd
from local_rag_backend.cli_commands.docs.docs_import_canonical import import_canonical_cmd
from local_rag_backend.cli_commands.docs.docs_ingest import ingest_cmd
from local_rag_backend.cli_commands.docs.docs_mutate import mutate_docs_cmd

__all__ = [
    "bootstrap_cmd",
    "import_canonical_cmd",
    "ingest_cmd",
    "mutate_docs_cmd",
]
