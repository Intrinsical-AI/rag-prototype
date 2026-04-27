"""Evaluation dataset and qrels parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_rag_backend.core.services.evaluation_models import (
    EvalDataset,
    EvalDoc,
    EvalQrel,
    EvalQuery,
)


def default_eval_dataset_path() -> Path:
    return Path(__file__).resolve().parents[4] / "datasets" / "rag_eval_v1.jsonl"


def _parse_schema_version(raw_value: Any, *, lineno: int) -> int:
    try:
        return int(raw_value or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid dataset line {lineno}: schema_version must be an integer."
        ) from exc


def _normalize_external_id(raw_value: object, *, lineno: int, field_name: str) -> str:
    external_id = str(raw_value).strip()
    if not external_id:
        raise ValueError(f"Invalid dataset line {lineno}: {field_name} must not be blank.")
    return external_id


def _parse_qrel_relevance(raw_value: object, *, lineno: int) -> int:
    try:
        if raw_value is None:
            relevance = 1
        elif isinstance(raw_value, bool):
            raise ValueError
        elif isinstance(raw_value, int):
            relevance = raw_value
        elif isinstance(raw_value, str):
            relevance = int(raw_value.strip())
        else:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid dataset line {lineno}: qrel relevance must be an integer."
        ) from exc
    if relevance < 0:
        raise ValueError(f"Invalid dataset line {lineno}: qrel relevance must be >= 0.")
    return relevance


def _parse_query_qrels(
    obj: dict[str, Any],
    *,
    lineno: int,
    schema_version: int,
) -> tuple[EvalQrel, ...]:
    raw_qrels = obj.get("qrels")
    if raw_qrels is not None:
        if schema_version != 2:
            raise ValueError(f"Invalid dataset line {lineno}: qrels require schema_version=2.")
        if not isinstance(raw_qrels, list):
            raise ValueError(f"Invalid dataset line {lineno}: qrels must be a list.")
        qrels: list[EvalQrel] = []
        seen: set[str] = set()
        for index, raw_item in enumerate(raw_qrels):
            if not isinstance(raw_item, dict):
                raise ValueError(
                    f"Invalid dataset line {lineno}: qrels[{index}] must be an object."
                )
            external_id = _normalize_external_id(
                raw_item.get("external_id"),
                lineno=lineno,
                field_name=f"qrels[{index}].external_id",
            )
            if external_id in seen:
                raise ValueError(
                    f"Invalid dataset line {lineno}: duplicate qrel external_id={external_id!r}."
                )
            seen.add(external_id)
            qrels.append(
                EvalQrel(
                    external_id=external_id,
                    relevance=_parse_qrel_relevance(raw_item.get("relevance", 1), lineno=lineno),
                )
            )
        if not any(qrel.relevance > 0 for qrel in qrels):
            raise ValueError(
                f"Invalid dataset line {lineno}: qrels must contain at least one positive relevance."
            )
        return tuple(qrels)

    rel = obj.get("relevant_external_ids") or []
    if not isinstance(rel, list) or not all(isinstance(x, str) for x in rel):
        raise ValueError(f"Invalid dataset line {lineno}: relevant_external_ids must be list[str].")
    normalized_rel = tuple(str(external_id).strip() for external_id in rel)
    if not normalized_rel:
        raise ValueError(f"Invalid dataset line {lineno}: relevant_external_ids must not be empty.")
    if any(not external_id for external_id in normalized_rel):
        raise ValueError(
            f"Invalid dataset line {lineno}: relevant_external_ids must not contain blank IDs."
        )
    if len(set(normalized_rel)) != len(normalized_rel):
        raise ValueError(f"Invalid dataset line {lineno}: duplicate relevant_external_ids.")
    return tuple(EvalQrel(external_id=external_id, relevance=1) for external_id in normalized_rel)


def qrels_for_query(query: EvalQuery) -> tuple[EvalQrel, ...]:
    if query.qrels:
        return query.qrels
    return tuple(
        EvalQrel(external_id=str(external_id).strip(), relevance=1)
        for external_id in query.relevant_external_ids
        if str(external_id).strip()
    )


def load_eval_dataset(path: str | Path | None = None) -> EvalDataset:
    content: str
    if path is None:
        p = default_eval_dataset_path()
        if not p.is_file():
            raise FileNotFoundError(
                f"Default eval dataset not found: {p}. Pass --dataset or configure "
                "eval_dataset_path in config.yaml."
            )
        content = p.read_text(encoding="utf-8")
    else:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Dataset not found: {p}")
        content = p.read_text(encoding="utf-8")

    dataset_id = "unknown"
    schema_version = 0
    docs: list[EvalDoc] = []
    queries: list[EvalQuery] = []
    known_doc_ids: set[str] = set()

    for lineno, line in enumerate(content.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid dataset line {lineno}: expected JSON object.")
        t = obj.get("type")
        if t == "meta":
            dataset_id = str(obj.get("dataset_id") or dataset_id)
            schema_version = _parse_schema_version(obj.get("schema_version"), lineno=lineno)
        elif t == "doc":
            external_id = _normalize_external_id(
                obj.get("external_id"), lineno=lineno, field_name="external_id"
            )
            if external_id in known_doc_ids:
                raise ValueError(
                    f"Invalid dataset line {lineno}: duplicate doc external_id={external_id!r}."
                )
            known_doc_ids.add(external_id)
            docs.append(
                EvalDoc(
                    external_id=external_id,
                    content=str(obj["content"]),
                    source_id=(
                        str(obj.get("source_id")) if obj.get("source_id") is not None else None
                    ),
                )
            )
        elif t == "query":
            qrels = _parse_query_qrels(obj, lineno=lineno, schema_version=schema_version)
            queries.append(
                EvalQuery(
                    query=str(obj["query"]),
                    relevant_external_ids=tuple(
                        qrel.external_id for qrel in qrels if qrel.relevance > 0
                    ),
                    qrels=qrels,
                )
            )
        else:
            raise ValueError(f"Invalid dataset line {lineno}: unknown type={t!r}")

    if schema_version not in {1, 2}:
        raise ValueError(f"Unsupported schema_version={schema_version} (expected 1 or 2).")
    if not docs:
        raise ValueError("Dataset contains no docs.")
    if not queries:
        raise ValueError("Dataset contains no queries.")
    for query in queries:
        missing_ids = sorted(
            {
                qrel.external_id
                for qrel in qrels_for_query(query)
                if qrel.external_id not in known_doc_ids
            }
        )
        if missing_ids:
            raise ValueError(
                "Dataset query references relevant_external_ids outside the corpus: "
                + ", ".join(missing_ids[:10])
            )

    return EvalDataset(
        dataset_id=str(dataset_id),
        schema_version=int(schema_version),
        docs=tuple(docs),
        queries=tuple(queries),
    )
