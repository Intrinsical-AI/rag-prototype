from __future__ import annotations

import pytest

from local_rag_backend.core.services.evaluation_models import EvalBenchmark
from local_rag_backend.infrastructure.evaluation import eval_dataset_from_ir_mappings


def test_eval_dataset_from_ir_mappings_converts_beir_like_records() -> None:
    dataset = eval_dataset_from_ir_mappings(
        benchmark=EvalBenchmark(
            benchmark_id="beir-fixture",
            adapter="beir",
            dataset_id="demo",
            split="test",
        ),
        corpus={
            "d1": {"title": "Alpha", "text": "first doc"},
            "d2": {"text": "second doc"},
        },
        queries={"q1": "alpha?"},
        qrels={"q1": {"d1": 2, "d2": 1}},
    )

    assert dataset.dataset_id == "beir:demo"
    assert dataset.schema_version == 2
    assert dataset.docs[0].content == "Alpha\n\nfirst doc"
    assert dataset.queries[0].qrels[0].relevance == 2


def test_eval_dataset_from_ir_mappings_rejects_missing_qrel_docs() -> None:
    with pytest.raises(ValueError, match="outside the corpus"):
        eval_dataset_from_ir_mappings(
            benchmark=EvalBenchmark(
                benchmark_id="beir-fixture",
                adapter="beir",
                dataset_id="demo",
            ),
            corpus={"d1": "doc"},
            queries={"q1": "alpha?"},
            qrels={"q1": {"missing": 1}},
        )
