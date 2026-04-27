"""Validation and parsing helpers for evaluation configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from local_rag_backend.core.services.evaluation_models import (
    EvalBatchSpec,
    EvalCompareConfig,
    EvalCompareSpec,
    EvalCompareThresholds,
    EvalRetrievalConfig,
    EvalRetrievalMode,
)

VALID_EVAL_RETRIEVAL_MODES = frozenset({"sparse", "dense", "dual", "hybrid"})


def parse_int_field(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got boolean.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value.strip())
    raise ValueError(f"{field_name} must be an integer.")


def parse_optional_int_field(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return parse_int_field(value, field_name=field_name)


def parse_optional_float_field(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, got boolean.")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise ValueError(f"{field_name} must be numeric.")


def parse_bool_field(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be boolean.")


def parse_eval_retrieval_mode(
    value: object | None,
    *,
    field_name: str = "retrieval_mode",
    default: EvalRetrievalMode | None = None,
) -> EvalRetrievalMode:
    raw = default if value is None else value
    normalized = str(raw).strip().lower()
    if normalized not in VALID_EVAL_RETRIEVAL_MODES:
        raise ValueError(f"{field_name} must be one of sparse, dense, dual, hybrid.")
    return cast("EvalRetrievalMode", normalized)


def validate_eval_retrieval_config(
    config: EvalRetrievalConfig,
    *,
    field_names: Mapping[str, str] | None = None,
) -> EvalRetrievalConfig:
    labels = field_names or {}
    mode = parse_eval_retrieval_mode(
        config.retrieval_mode,
        field_name=labels.get("retrieval_mode", "retrieval_mode"),
    )
    if int(config.k) <= 0:
        raise ValueError(f"{labels.get('k', 'k')} must be positive")
    if config.candidate_k is not None and mode != "dense":
        raise ValueError(
            f"{labels.get('candidate_k', 'candidate_k')} is supported only with "
            "retrieval_mode=dense"
        )
    if config.dual_candidate_k is not None and mode != "dual":
        raise ValueError(
            f"{labels.get('dual_candidate_k', 'dual_candidate_k')} is supported only with "
            "retrieval_mode=dual"
        )
    if config.hybrid_alpha is not None and mode != "hybrid":
        raise ValueError(
            f"{labels.get('hybrid_alpha', 'hybrid_alpha')} is supported only with "
            "retrieval_mode=hybrid"
        )
    if config.candidate_k is not None and int(config.candidate_k) <= 0:
        raise ValueError(f"{labels.get('candidate_k', 'candidate_k')} must be positive")
    if config.dual_candidate_k is not None and int(config.dual_candidate_k) <= 0:
        raise ValueError(
            f"{labels.get('dual_candidate_k', 'dual_candidate_k')} must be positive"
        )
    if config.hybrid_alpha is not None and not 0.0 <= float(config.hybrid_alpha) <= 1.0:
        raise ValueError(
            f"{labels.get('hybrid_alpha', 'hybrid_alpha')} must be between 0.0 and 1.0"
        )
    return EvalRetrievalConfig(
        retrieval_mode=mode,
        k=int(config.k),
        candidate_k=(int(config.candidate_k) if config.candidate_k is not None else None),
        dual_candidate_k=(
            int(config.dual_candidate_k) if config.dual_candidate_k is not None else None
        ),
        hybrid_alpha=(float(config.hybrid_alpha) if config.hybrid_alpha is not None else None),
        reranker_enabled=bool(config.reranker_enabled),
    )


def build_eval_retrieval_config(
    *,
    retrieval_mode: object | None = None,
    k: object = 3,
    candidate_k: object | None = None,
    dual_candidate_k: object | None = None,
    hybrid_alpha: object | None = None,
    reranker_enabled: object = False,
    default_mode: EvalRetrievalMode = "sparse",
    field_names: Mapping[str, str] | None = None,
) -> EvalRetrievalConfig:
    labels = field_names or {}
    parsed = EvalRetrievalConfig(
        retrieval_mode=parse_eval_retrieval_mode(
            retrieval_mode,
            field_name=labels.get("retrieval_mode", "retrieval_mode"),
            default=default_mode,
        ),
        k=parse_int_field(k, field_name=labels.get("k", "k")),
        candidate_k=parse_optional_int_field(
            candidate_k,
            field_name=labels.get("candidate_k", "candidate_k"),
        ),
        dual_candidate_k=parse_optional_int_field(
            dual_candidate_k,
            field_name=labels.get("dual_candidate_k", "dual_candidate_k"),
        ),
        hybrid_alpha=parse_optional_float_field(
            hybrid_alpha,
            field_name=labels.get("hybrid_alpha", "hybrid_alpha"),
        ),
        reranker_enabled=(
            reranker_enabled
            if isinstance(reranker_enabled, bool)
            else parse_bool_field(
                reranker_enabled,
                field_name=labels.get("reranker_enabled", "reranker_enabled"),
            )
        ),
    )
    return validate_eval_retrieval_config(parsed, field_names=labels)


def eval_compare_config_to_retrieval_config(
    config: EvalCompareConfig,
    *,
    k: int,
) -> EvalRetrievalConfig:
    return validate_eval_retrieval_config(
        EvalRetrievalConfig(
            retrieval_mode=config.retrieval_mode,
            k=int(k),
            candidate_k=config.candidate_k,
            dual_candidate_k=config.dual_candidate_k,
            hybrid_alpha=config.hybrid_alpha,
            reranker_enabled=config.reranker_enabled,
        )
    )


def parse_eval_compare_config(raw: object, *, field_name: str) -> EvalCompareConfig:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be an object.")
    mode = parse_eval_retrieval_mode(
        raw.get("retrieval_mode"),
        field_name=f"{field_name}.retrieval_mode",
    )
    config = EvalCompareConfig(
        retrieval_mode=mode,
        candidate_k=parse_optional_int_field(
            raw.get("candidate_k"),
            field_name=f"{field_name}.candidate_k",
        ),
        dual_candidate_k=parse_optional_int_field(
            raw.get("dual_candidate_k"),
            field_name=f"{field_name}.dual_candidate_k",
        ),
        hybrid_alpha=parse_optional_float_field(
            raw.get("hybrid_alpha"),
            field_name=f"{field_name}.hybrid_alpha",
        ),
        reranker_enabled=(
            bool(raw.get("reranker_enabled", False))
            if isinstance(raw.get("reranker_enabled", False), bool)
            else parse_bool_field(
                raw.get("reranker_enabled"),
                field_name=f"{field_name}.reranker_enabled",
            )
        ),
    )
    validate_eval_retrieval_config(
        EvalRetrievalConfig(
            retrieval_mode=config.retrieval_mode,
            k=1,
            candidate_k=config.candidate_k,
            dual_candidate_k=config.dual_candidate_k,
            hybrid_alpha=config.hybrid_alpha,
            reranker_enabled=config.reranker_enabled,
        ),
        field_names={
            "retrieval_mode": f"{field_name}.retrieval_mode",
            "candidate_k": f"{field_name}.candidate_k",
            "dual_candidate_k": f"{field_name}.dual_candidate_k",
            "hybrid_alpha": f"{field_name}.hybrid_alpha",
        },
    )
    return config


def parse_eval_compare_thresholds(raw: object | None) -> EvalCompareThresholds:
    if raw is None:
        return EvalCompareThresholds()
    if not isinstance(raw, Mapping):
        raise ValueError("thresholds must be an object.")
    return EvalCompareThresholds(
        min_delta_ndcg=float(raw.get("min_delta_ndcg", 0.0)),
        min_delta_map=float(raw.get("min_delta_map", 0.0)),
        min_delta_mrr=float(raw.get("min_delta_mrr", 0.0)),
        max_regression_precision=float(raw.get("max_regression_precision", 0.0)),
        max_regression_recall=float(raw.get("max_regression_recall", 0.0)),
    )


def parse_eval_compare_spec(payload: object) -> EvalCompareSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("Compare spec must be a JSON object.")
    if "baseline" not in payload:
        raise ValueError("Compare spec is missing required field 'baseline'.")
    if "candidate" not in payload:
        raise ValueError("Compare spec is missing required field 'candidate'.")
    k = parse_int_field(payload.get("k", 3), field_name="k")
    if k <= 0:
        raise ValueError("k must be positive")
    max_queries = parse_optional_int_field(payload.get("max_queries"), field_name="max_queries")
    if max_queries is not None and max_queries < 0:
        raise ValueError("max_queries must be >= 0")
    baseline = parse_eval_compare_config(payload["baseline"], field_name="baseline")
    candidate = parse_eval_compare_config(payload["candidate"], field_name="candidate")
    return EvalCompareSpec(
        baseline=baseline,
        candidate=candidate,
        thresholds=parse_eval_compare_thresholds(payload.get("thresholds")),
        k=k,
        max_queries=max_queries,
    )


def parse_eval_batch_spec(raw: Mapping[str, object], *, index: int) -> EvalBatchSpec:
    required_fields = ("name", "retrieval_mode", "k")
    for field_name in required_fields:
        if field_name not in raw:
            raise ValueError(f"Spec at index {index} is missing required field {field_name!r}.")
    config = build_eval_retrieval_config(
        retrieval_mode=raw["retrieval_mode"],
        k=raw["k"],
        candidate_k=raw.get("candidate_k"),
        dual_candidate_k=raw.get("dual_candidate_k"),
        hybrid_alpha=raw.get("hybrid_alpha"),
        reranker_enabled=raw.get("reranker_enabled", False),
        field_names={
            "retrieval_mode": f"specs[{index}].retrieval_mode",
            "k": f"specs[{index}].k",
            "candidate_k": f"specs[{index}].candidate_k",
            "dual_candidate_k": f"specs[{index}].dual_candidate_k",
            "hybrid_alpha": f"specs[{index}].hybrid_alpha",
            "reranker_enabled": f"specs[{index}].reranker_enabled",
        },
    )
    return EvalBatchSpec(
        name=str(raw["name"]),
        retrieval_mode=config.retrieval_mode,
        k=config.k,
        candidate_k=config.candidate_k,
        dual_candidate_k=config.dual_candidate_k,
        hybrid_alpha=config.hybrid_alpha,
        reranker_enabled=config.reranker_enabled,
        json_out=(str(raw["json_out"]).strip() if raw.get("json_out") is not None else None),
        run_out=(str(raw["run_out"]).strip() if raw.get("run_out") is not None else None),
        report_out=(
            str(raw["report_out"]).strip() if raw.get("report_out") is not None else None
        ),
        anomalies_out=(
            str(raw["anomalies_out"]).strip()
            if raw.get("anomalies_out") is not None
            else None
        ),
    )
