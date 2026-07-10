from __future__ import annotations

import pytest
from click.testing import CliRunner
from support.repogpt_fixture import REPOGPT_CLI_AVAILABLE, emit_repogpt_code_units

if not REPOGPT_CLI_AVAILABLE:
    pytest.skip(
        "cross-repo RepoGPT checkout not available",
        allow_module_level=True,
    )

from local_rag_backend.cli import cli
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever
from local_rag_backend.settings import settings


def test_repogpt_emit_code_units_imports_and_retrieves_by_queryable_metadata(
    in_memory_sqlite, tmp_path, monkeypatch
):
    _ = in_memory_sqlite
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "data_dir", data_dir, raising=False)

    payload_path = tmp_path / "repogpt_code_units.json"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "sample.py").write_text(
        "class Demo:\n"
        "    def method(self, value: int) -> int:\n"
        "        return value + 1\n"
        "\n"
        "def helper(name: str = 'world') -> str:\n"
        "    return name\n",
        encoding="utf-8",
    )
    payload = emit_repogpt_code_units(payload_path=payload_path, repo_path=repo_path)

    assert payload["schema_version"] == "4"
    assert payload["replace_scope"] is True

    result = CliRunner().invoke(cli, ["import-canonical", "--json", str(payload_path)])
    assert result.exit_code == 0, result.output

    repo = SqlDocumentStorage()
    docs = list(repo.get_all_documents())
    assert docs

    retriever = LocalSplitSearchRetriever(doc_repo=repo)
    retrieval = retriever.retrieve(
        RetrievalRequest(
            query="helper",
            top_k=1,
            mode="sparse",
            filters=(
                RetrievalFilter(field="metadata.path", values=("sample.py",)),
                RetrievalFilter(field="metadata.unit_type", values=("function",)),
            ),
        )
    )

    assert retrieval.items
    document = retrieval.items[0].document
    metadata = dict(document.metadata or {})
    repo_key = str(payload["repo_key"])
    assert document.external_id == f"repogpt:{repo_key}:sample.py:function:helper"
    assert metadata["path"] == "sample.py"
    assert metadata["unit_type"] == "function"
    assert metadata["repo_key"] == repo_key
    assert str(metadata["content_hash"]).strip()
