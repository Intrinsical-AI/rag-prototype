"""Benchmark dataset adapters for IR-style corpora."""

from __future__ import annotations

from collections.abc import Mapping

from local_rag_backend.core.services.evaluation_models import (
    EvalBenchmark,
    EvalDataset,
    EvalDoc,
    EvalQrel,
    EvalQuery,
)


def _document_content(raw_doc: object) -> str:
    if isinstance(raw_doc, str):
        return raw_doc
    if isinstance(raw_doc, Mapping):
        title = str(raw_doc.get("title") or "").strip()
        text = str(raw_doc.get("text") or raw_doc.get("content") or "").strip()
        if title and text:
            return f"{title}\n\n{text}"
        return title or text
    return str(raw_doc)


def eval_dataset_from_ir_mappings(
    *,
    benchmark: EvalBenchmark,
    corpus: Mapping[str, object],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
) -> EvalDataset:
    """Convert BEIR/MTEB-like in-memory corpus/query/qrels mappings to EvalDataset."""
    docs = tuple(
        EvalDoc(
            external_id=str(doc_id),
            content=_document_content(raw_doc),
            source_id=benchmark.benchmark_id,
        )
        for doc_id, raw_doc in corpus.items()
    )
    known_doc_ids = {doc.external_id for doc in docs}
    eval_queries: list[EvalQuery] = []
    for query_id, query_text in queries.items():
        raw_qrels = qrels.get(str(query_id), {})
        parsed_qrels = tuple(
            EvalQrel(external_id=str(doc_id), relevance=int(relevance))
            for doc_id, relevance in raw_qrels.items()
            if int(relevance) > 0
        )
        if not parsed_qrels:
            continue
        missing = sorted(
            qrel.external_id for qrel in parsed_qrels if qrel.external_id not in known_doc_ids
        )
        if missing:
            raise ValueError(
                "Benchmark qrels reference documents outside the corpus: " + ", ".join(missing[:10])
            )
        eval_queries.append(
            EvalQuery(
                query=str(query_text),
                relevant_external_ids=tuple(qrel.external_id for qrel in parsed_qrels),
                qrels=parsed_qrels,
            )
        )
    if not docs:
        raise ValueError("Benchmark corpus contains no docs.")
    if not eval_queries:
        raise ValueError("Benchmark qrels contain no positive relevance judgments.")
    dataset_id = f"{benchmark.adapter}:{benchmark.dataset_id}"
    return EvalDataset(
        dataset_id=dataset_id,
        schema_version=2,
        docs=docs,
        queries=tuple(eval_queries),
    )
