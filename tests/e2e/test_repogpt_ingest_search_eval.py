from __future__ import annotations

import copy
import json
from pathlib import Path

from click.testing import CliRunner
from support.repogpt_fixture import REPOGPT_FIXTURE_REPO, emit_repogpt_code_units

from local_rag_backend.cli import cli
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.core.services.evaluation import load_eval_dataset
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever
from local_rag_backend.settings import settings

DATASET_PATH = Path(__file__).resolve().parents[2] / "datasets" / "repogpt_rag_eval_v1.jsonl"


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


def test_repogpt_fixture_import_search_eval_and_scope_sync(
    in_memory_sqlite, tmp_path, monkeypatch
) -> None:
    _ = in_memory_sqlite
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "search_backend", "local_split", raising=False)
    monkeypatch.setattr(settings, "persistence_backend", "local_split", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "data_dir", data_dir, raising=False)

    payload_path = tmp_path / "repogpt_eval_code_units.json"
    payload = emit_repogpt_code_units(payload_path=payload_path, repo_path=REPOGPT_FIXTURE_REPO)

    assert payload["schema_version"] == "3"
    assert payload["replace_scope"] is True
    repo_key = str(payload["repo_key"])

    dataset = load_eval_dataset(DATASET_PATH)
    emitted_external_ids = {
        str(document["external_id"])
        for document in list(payload["documents"])  # type: ignore[index]
    }
    dataset_external_ids = {doc.external_id for doc in dataset.docs}
    assert dataset_external_ids <= emitted_external_ids

    runner = CliRunner()
    result = runner.invoke(cli, ["import-canonical", "--json", str(payload_path)])
    assert result.exit_code == 0, result.output

    repo = SqlDocumentStorage()
    docs = list(repo.get_all_documents())
    assert len(docs) == len(list(payload["documents"]))  # type: ignore[arg-type]
    retriever = LocalSplitSearchRetriever(doc_repo=repo)

    helper_id = _retrieved_external_id(
        retriever=retriever,
        query="helper",
        filters=(
            RetrievalFilter(field="path", values=("sample.py",)),
            RetrievalFilter(field="unit_type", values=("function",)),
        ),
    )
    assert helper_id == f"repogpt:{repo_key}:sample.py:function:helper"

    validate_token_id = _retrieved_external_id(
        retriever=retriever,
        query="validate token",
        filters=(RetrievalFilter(field="metadata.symbol", values=("validate_token",)),),
    )
    assert validate_token_id == f"repogpt:{repo_key}:auth.py:function:validate_token"

    assert (
        _retrieved_external_id(
            retriever=retriever,
            query="authorization header bearer token",
        )
        == f"repogpt:{repo_key}:auth.py:function:extract_bearer_token"
    )
    assert (
        _retrieved_external_id(
            retriever=retriever,
            query="build the api client configured for a base URL",
        )
        == f"repogpt:{repo_key}:client.py:function:build_api_client"
    )
    assert (
        _retrieved_external_id(
            retriever=retriever,
            query="load configuration values from environment",
        )
        == f"repogpt:{repo_key}:config.py:function:load_config"
    )

    docs_by_external_id = {doc.external_id: doc for doc in docs}
    helper_doc = docs_by_external_id[f"repogpt:{repo_key}:sample.py:function:helper"]
    helper_metadata = dict(helper_doc.metadata or {})
    assert helper_metadata["path"] == "sample.py"
    assert helper_metadata["unit_type"] == "function"
    assert helper_metadata["symbol"] == "helper"
    assert helper_metadata["repo_key"] == repo_key
    assert str(helper_metadata["content_hash"]).strip()

    second_result = runner.invoke(cli, ["import-canonical", "--json", str(payload_path)])
    assert second_result.exit_code == 0, second_result.output
    assert len(list(repo.get_all_documents())) == len(docs)

    reduced_payload = copy.deepcopy(payload)
    reduced_payload["snapshot_id"] = f"{payload['snapshot_id']}-reduced"
    reduced_payload["documents"] = [
        document
        for document in list(payload["documents"])  # type: ignore[index]
        if str(document["external_id"]).endswith(":sample.py:function:helper")
        or str(document["external_id"]).endswith(":client.py:function:build_api_client")
    ]
    reduced_payload_path = tmp_path / "repogpt_eval_code_units_reduced.json"
    reduced_payload_path.write_text(
        json.dumps(reduced_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    reduced_result = runner.invoke(cli, ["import-canonical", "--json", str(reduced_payload_path)])
    assert reduced_result.exit_code == 0, reduced_result.output

    reduced_docs = list(repo.get_all_documents())
    reduced_external_ids = {doc.external_id for doc in reduced_docs}
    assert reduced_external_ids == {
        f"repogpt:{repo_key}:sample.py:function:helper",
        f"repogpt:{repo_key}:client.py:function:build_api_client",
    }
