from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from support.vuln_pilot_fixture import SYNERGY_ROOT, VULN_PILOT_PREPARED, load_vulns_ingest_module

if (
    not VULN_PILOT_PREPARED.exists()
    or not (SYNERGY_ROOT / "scripts" / "vulns_ingest_rag.py").exists()
):
    pytest.skip(
        "cross-repo vuln pilot data not available (synergy monorepo layout required)",
        allow_module_level=True,
    )

from local_rag_backend.cli import cli
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.core.services.evaluation import load_eval_dataset
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever
from local_rag_backend.settings import settings

DATASET_PATH = Path(__file__).resolve().parents[2] / "datasets" / "vuln_pilot_rag_eval_v1.jsonl"


def _retrieved_external_id(
    *,
    retriever: LocalSplitSearchRetriever,
    query: str,
    filters: tuple[RetrievalFilter, ...] = (),
) -> str:
    retrieval = retriever.retrieve(
        RetrievalRequest(
            query=query,
            top_k=1,
            mode="sparse",
            filters=filters,
        )
    )
    assert retrieval.items
    return retrieval.items[0].document.external_id


def test_vuln_pilot_import_search_eval_and_scope_sync(
    in_memory_sqlite, tmp_path, monkeypatch
) -> None:
    _ = in_memory_sqlite
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "search_backend", "local_split", raising=False)
    monkeypatch.setattr(settings, "persistence_backend", "local_split", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "data_dir", data_dir, raising=False)

    module = load_vulns_ingest_module()
    records = module.load_vuln_records(VULN_PILOT_PREPARED)
    payloads = module.build_canonical_payloads(
        records,
        dataset_name="pilot_small_v1",
        scope="vuln-triage:pilot-small-v1",
        snapshot_id="pilot-small-v1",
    )
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["replace_scope"] is True
    assert payload["scope"] == "vuln-triage:pilot-small-v1"
    assert payload["documents"][0]["source_id"] == "vuln-source:circl_vulnerability_cwe_patch"

    dataset = load_eval_dataset(DATASET_PATH)
    emitted_external_ids = {
        str(document["external_id"])
        for document in list(payload["documents"])  # type: ignore[index]
    }
    dataset_external_ids = {doc.external_id for doc in dataset.docs}
    assert dataset_external_ids <= emitted_external_ids

    payload_path = tmp_path / "vuln_pilot_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["import-canonical", "--json", str(payload_path)])
    assert result.exit_code == 0, result.output

    repo = SqlDocumentStorage()
    docs = list(repo.get_all_documents())
    assert len(docs) == len(list(payload["documents"]))  # type: ignore[arg-type]
    retriever = LocalSplitSearchRetriever(doc_repo=repo)

    assert (
        _retrieved_external_id(
            retriever=retriever,
            query="malicious MCP server command injection",
            filters=(RetrievalFilter(field="metadata.cwe_id", values=("CWE-78",)),),
        )
        == "vuln:CVE-2025-54074"
    )
    assert (
        _retrieved_external_id(
            retriever=retriever,
            query="mkdtemp windows temporary directory permissions",
            filters=(RetrievalFilter(field="metadata.language", values=("python",)),),
        )
        == "vuln:CVE-2024-4030"
    )
    assert (
        _retrieved_external_id(
            retriever=retriever,
            query="missing authentication rdiffweb",
            filters=(RetrievalFilter(field="metadata.severity", values=("CRITICAL",)),),
        )
        == "vuln:CVE-2022-3327"
    )
    assert (
        _retrieved_external_id(
            retriever=retriever,
            query="temporary directory permissions on windows",
            filters=(RetrievalFilter(field="metadata.repo", values=("python/cpython",)),),
        )
        == "vuln:CVE-2024-4030"
    )
    assert (
        _retrieved_external_id(
            retriever=retriever,
            query="download service raw requests get safe_requests ssrf",
        )
        == "vuln:CVE-2025-67743"
    )
    assert (
        _retrieved_external_id(
            retriever=retriever,
            query="waitress request smuggling behind proxy",
        )
        == "vuln:CVE-2022-24761"
    )

    docs_by_external_id = {doc.external_id: doc for doc in docs}
    cwe78_doc = docs_by_external_id["vuln:CVE-2025-54074"]
    cwe78_metadata = dict(cwe78_doc.metadata or {})
    assert cwe78_metadata["cwe_id"] == "CWE-78"
    assert cwe78_metadata["language"] == "typescript"
    assert cwe78_metadata["severity"] == "CRITICAL"
    assert cwe78_metadata["repo"] == "CherryHQ/cherry-studio"

    second_result = runner.invoke(cli, ["import-canonical", "--json", str(payload_path)])
    assert second_result.exit_code == 0, second_result.output
    assert len(list(repo.get_all_documents())) == len(docs)

    reduced_payload = copy.deepcopy(payload)
    reduced_payload["snapshot_id"] = "pilot-small-v1-reduced"
    reduced_payload["documents"] = [
        document
        for document in list(payload["documents"])  # type: ignore[index]
        if str(document["external_id"]) in {"vuln:CVE-2022-3327", "vuln:CVE-2025-54074"}
    ]
    reduced_payload_path = tmp_path / "vuln_pilot_payload_reduced.json"
    reduced_payload_path.write_text(
        json.dumps(reduced_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    reduced_result = runner.invoke(cli, ["import-canonical", "--json", str(reduced_payload_path)])
    assert reduced_result.exit_code == 0, reduced_result.output

    reduced_docs = list(repo.get_all_documents())
    reduced_external_ids = {doc.external_id for doc in reduced_docs}
    assert reduced_external_ids == {"vuln:CVE-2022-3327", "vuln:CVE-2025-54074"}
