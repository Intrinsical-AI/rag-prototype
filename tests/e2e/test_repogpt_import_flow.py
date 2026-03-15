from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever
from local_rag_backend.settings import settings

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SYNERGY_ROOT = WORKSPACE_ROOT / "synergy"
REPOGPT_ROOT = WORKSPACE_ROOT / "RepoGPT"
REPOGPT_SRC = REPOGPT_ROOT / "src"
pytest.importorskip("structlog")
if str(REPOGPT_SRC) not in sys.path:
    sys.path.insert(0, str(REPOGPT_SRC))


def _load_repogpt_types() -> tuple[type[object], type[object], type[object], type[object]]:
    publisher_module = importlib.import_module("repogpt.adapters.publisher.code_units_publisher")
    models_module = importlib.import_module("repogpt.models")
    return (
        publisher_module.CodeUnitsPublisher,
        models_module.AnalysisConf,
        models_module.CodeNode,
        models_module.PipelineResult,
    )


def _write_repogpt_code_units(payload_path: Path) -> dict[str, object]:
    CodeUnitsPublisher, AnalysisConf, CodeNode, PipelineResult = _load_repogpt_types()
    sample = payload_path.parent / "sample.py"
    sample.write_text(
        "class Demo:\n"
        "    def method(self, value: int) -> int:\n"
        "        return value + 1\n"
        "\n"
        "def helper(name: str = 'world') -> str:\n"
        "    return name\n",
        encoding="utf-8",
    )
    method = CodeNode(
        id="method-1",
        type="method",
        name="method",
        language="py",
        path="sample.py",
        start_line=2,
        end_line=3,
        parent_id="class-1",
    )
    klass = CodeNode(
        id="class-1",
        type="class",
        name="Demo",
        language="py",
        path="sample.py",
        start_line=1,
        end_line=3,
        children=[method],
    )
    helper = CodeNode(
        id="function-1",
        type="function",
        name="helper",
        language="py",
        path="sample.py",
        start_line=5,
        end_line=6,
    )
    root = CodeNode(
        id="module-1",
        type="module",
        name="sample",
        language="py",
        path="sample.py",
        start_line=1,
        end_line=6,
        children=[klass, helper],
    )
    CodeUnitsPublisher().publish(
        [
            PipelineResult(
                path=sample,
                language="py",
                root=root,
                file_info={
                    "relative_path": "sample.py",
                    "size": sample.stat().st_size,
                    "sha256": "fixture-sha",
                },
                content=sample.read_text(encoding="utf-8"),
            )
        ],
        AnalysisConf(repo_path=payload_path.parent, output=payload_path, emit_kind="code-units"),
    )
    return json.loads(payload_path.read_text(encoding="utf-8"))


def test_repogpt_emit_code_units_imports_and_retrieves_by_queryable_metadata(
    in_memory_sqlite, tmp_path, monkeypatch
):
    _ = in_memory_sqlite
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "data_dir", data_dir, raising=False)

    payload_path = tmp_path / "repogpt_code_units.json"
    payload = _write_repogpt_code_units(payload_path)

    assert payload["schema_version"] == "3"
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
                RetrievalFilter(field="path", values=("sample.py",)),
                RetrievalFilter(field="unit_type", values=("function",)),
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
