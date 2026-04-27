from __future__ import annotations

import sys

import pytest

from local_rag_backend.core.services.evaluation import load_eval_dataset, run_retrieval_eval
from local_rag_backend.infrastructure.evaluation import (
    EvalAdapterUnavailableError,
    RagasEvaluationAdapter,
)


def test_ragas_adapter_builds_core_report_payload_without_importing_ragas() -> None:
    ds = load_eval_dataset()
    result = run_retrieval_eval(
        dataset=ds,
        retrieve_external_ids=lambda _query, _top_k: ("doc:paris",),
        retrieval_mode="sparse",
        k=1,
        max_queries=1,
    )

    payload = RagasEvaluationAdapter().build_report_payload(result)

    assert payload["adapter"] == "ragas"
    assert payload["type"] == "eval_report"


def test_ragas_adapter_reports_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ragas", None)

    with pytest.raises(EvalAdapterUnavailableError, match="installing ragas"):
        RagasEvaluationAdapter().ensure_available()
