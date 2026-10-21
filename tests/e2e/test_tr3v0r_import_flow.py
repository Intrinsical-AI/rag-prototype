from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever
from local_rag_backend.settings import get_settings


def test_tr3v0r_v2_import_preserves_sensor_ids_and_retrieves_by_membership(
    in_memory_sqlite, tmp_path, monkeypatch
):
    settings = get_settings()
    monkeypatch.setattr(settings, "retrieval_mode", "sparse")
    raw = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "tr3v0r_canonical_v2.json").read_text()
    )
    path = tmp_path / "tr3v0r.json"
    path.write_text(json.dumps(raw))
    runner = CliRunner()
    first = runner.invoke(cli, ["import-canonical", "--json", str(path)])
    assert first.exit_code == 0, first.output
    second = runner.invoke(cli, ["import-canonical", "--json", str(path)])
    assert second.exit_code == 0, second.output
    assert "unchanged=4" in second.output
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    assert len(list(repo.get_all_documents())) == 4
    retriever = LocalSplitSearchRetriever(doc_repo=repo)
    result = retriever.retrieve(
        RetrievalRequest(
            query="sensor",
            top_k=1,
            mode="sparse",
            filters=(
                RetrievalFilter(field="metadata.doc_type", values=("net_node",)),
                RetrievalFilter(field="metadata.sensor_ids", values=("sensor",)),
            ),
        )
    )
    assert result.items
    assert result.items[0].document.metadata["sensor_ids"] == ["sensor"]
