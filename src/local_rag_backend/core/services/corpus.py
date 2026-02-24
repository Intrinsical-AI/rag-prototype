"""
Helpers around the document corpus used by retrievers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_rag_backend.core.ports import DocumentRepoPort


def get_corpus_and_ids(doc_repo: DocumentRepoPort) -> tuple[list[str], list[int]]:
    """Fetch all documents from a repository and separate contents from IDs."""
    docs = doc_repo.get_all_documents()
    return [d.content for d in docs], [d.id for d in docs]
