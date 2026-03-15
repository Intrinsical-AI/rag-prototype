from __future__ import annotations

import json
from pathlib import Path

import click

from local_rag_backend.cli_commands.runtime import get_cli_container


@click.command("eval")
@click.option(
    "--dataset",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Path to a JSONL eval dataset. If omitted, uses datasets/rag_eval_v1.jsonl "
        "(or RAG_EVAL_DATASET_PATH)."
    ),
)
@click.option(
    "--retrieval-mode",
    type=click.Choice(["sparse", "dense", "dual", "hybrid"], case_sensitive=False),
    default="sparse",
    show_default=True,
)
@click.option("--k", type=int, default=3, show_default=True)
@click.option(
    "--candidate-k",
    type=click.IntRange(1),
    default=None,
    help="Optional dense candidate count override.",
)
@click.option(
    "--dual-candidate-k",
    type=click.IntRange(1),
    default=None,
    help="Optional sparse candidate pool override for dual retrieval.",
)
@click.option(
    "--hybrid-alpha",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Optional sparse weight override for hybrid retrieval.",
)
@click.option("--max-queries", type=int, default=None, help="Evaluate only the first N queries.")
@click.option("--reranker/--no-reranker", default=False, show_default=True)
@click.option("--fail-below-ndcg", type=float, default=1.0, show_default=True)
@click.option("--fail-below-map", type=float, default=0.9, show_default=True)
@click.option("--fail-below-mrr", type=float, default=0.9, show_default=True)
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write eval results as JSON.",
)
def eval_cmd(
    dataset: Path | None,
    retrieval_mode: str,
    k: int,
    candidate_k: int | None,
    dual_candidate_k: int | None,
    hybrid_alpha: float | None,
    max_queries: int | None,
    reranker: bool,
    fail_below_ndcg: float,
    fail_below_map: float,
    fail_below_mrr: float,
    json_out: Path | None,
) -> None:
    """Offline IR evaluation with standard retrieval metrics."""
    try:
        from local_rag_backend.core.services.evaluation import (
            eval_result_to_json,
            format_eval_result,
            load_eval_dataset,
        )
        from local_rag_backend.core.use_cases.evaluation import run_retrieval_eval

        container = get_cli_container()
        eval_bundle = container.build_eval_execution_bundle()
        ds = load_eval_dataset(dataset)
        res = run_retrieval_eval(
            dataset=ds,
            eval_storage_port=eval_bundle.eval_storage_port,
            eval_retriever_factory_port=eval_bundle.eval_retriever_factory_port,
            retrieval_mode=retrieval_mode,
            k=k,
            candidate_k=candidate_k,
            reranker_enabled=bool(reranker),
            dual_candidate_k=dual_candidate_k,
            hybrid_alpha=hybrid_alpha,
            reranker_candidate_k=eval_bundle.reranker_candidate_k,
            reranker_strategy=eval_bundle.reranker_strategy,
            max_queries=max_queries,
        )
        if json_out is not None:
            json_out.write_text(json.dumps(eval_result_to_json(res)), encoding="utf-8")

        failures: list[str] = []
        if res.ndcg_at_k < float(fail_below_ndcg):
            failures.append(f"nDCG@{k}={res.ndcg_at_k:.3f} (min {fail_below_ndcg})")
        if res.map_at_k < float(fail_below_map):
            failures.append(f"MAP@{k}={res.map_at_k:.3f} (min {fail_below_map})")
        if res.mrr_at_k < float(fail_below_mrr):
            failures.append(f"MRR@{k}={res.mrr_at_k:.3f} (min {fail_below_mrr})")

        if failures:
            click.echo(
                "[ERROR] Eval regression: " + format_eval_result(res) + " | " + "; ".join(failures),
                err=True,
            )
            raise SystemExit(1)
        click.echo(format_eval_result(res))
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"[ERROR] Error evaluating dataset: {e}", err=True)
        raise SystemExit(1)


@click.command("eval-compare")
@click.option(
    "--dataset",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Path to a JSONL eval dataset. If omitted, uses datasets/rag_eval_v1.jsonl "
        "(or RAG_EVAL_DATASET_PATH)."
    ),
)
@click.option("--k", type=int, default=3, show_default=True)
@click.option(
    "--baseline-mode",
    type=click.Choice(["sparse", "dense", "dual", "hybrid"], case_sensitive=False),
    default="sparse",
    show_default=True,
)
@click.option("--baseline-candidate-k", type=click.IntRange(1), default=None)
@click.option("--baseline-dual-candidate-k", type=click.IntRange(1), default=None)
@click.option("--baseline-hybrid-alpha", type=click.FloatRange(0.0, 1.0), default=None)
@click.option(
    "--baseline-reranker/--no-baseline-reranker",
    default=False,
    show_default=True,
)
@click.option(
    "--candidate-mode",
    type=click.Choice(["sparse", "dense", "dual", "hybrid"], case_sensitive=False),
    required=True,
)
@click.option("--candidate-candidate-k", type=click.IntRange(1), default=None)
@click.option("--candidate-dual-candidate-k", type=click.IntRange(1), default=None)
@click.option("--candidate-hybrid-alpha", type=click.FloatRange(0.0, 1.0), default=None)
@click.option(
    "--candidate-reranker/--no-candidate-reranker",
    default=False,
    show_default=True,
)
@click.option("--max-queries", type=int, default=None, help="Evaluate only the first N queries.")
@click.option("--min-delta-ndcg", type=float, default=0.0, show_default=True)
@click.option("--min-delta-map", type=float, default=0.0, show_default=True)
@click.option("--min-delta-mrr", type=float, default=0.0, show_default=True)
@click.option("--max-regression-precision", type=float, default=0.0, show_default=True)
@click.option("--max-regression-recall", type=float, default=0.0, show_default=True)
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write compare results as JSON.",
)
def eval_compare_cmd(
    dataset: Path | None,
    k: int,
    baseline_mode: str,
    baseline_candidate_k: int | None,
    baseline_dual_candidate_k: int | None,
    baseline_hybrid_alpha: float | None,
    baseline_reranker: bool,
    candidate_mode: str,
    candidate_candidate_k: int | None,
    candidate_dual_candidate_k: int | None,
    candidate_hybrid_alpha: float | None,
    candidate_reranker: bool,
    max_queries: int | None,
    min_delta_ndcg: float,
    min_delta_map: float,
    min_delta_mrr: float,
    max_regression_precision: float,
    max_regression_recall: float,
    json_out: Path | None,
) -> None:
    """Compare baseline and candidate retrieval configs and fail on placebo improvements."""
    try:
        from local_rag_backend.core.services.evaluation import (
            eval_compare_result_to_json,
            format_eval_compare_result,
            load_eval_dataset,
        )
        from local_rag_backend.core.services.types import EvalCompareConfig
        from local_rag_backend.core.use_cases.evaluation import compare_retrieval_eval

        container = get_cli_container()
        eval_bundle = container.build_eval_execution_bundle()
        ds = load_eval_dataset(dataset)
        result = compare_retrieval_eval(
            dataset=ds,
            eval_storage_port=eval_bundle.eval_storage_port,
            eval_retriever_factory_port=eval_bundle.eval_retriever_factory_port,
            baseline=EvalCompareConfig(
                retrieval_mode=baseline_mode,
                candidate_k=baseline_candidate_k,
                dual_candidate_k=baseline_dual_candidate_k,
                hybrid_alpha=baseline_hybrid_alpha,
                reranker_enabled=bool(baseline_reranker),
            ),
            candidate=EvalCompareConfig(
                retrieval_mode=candidate_mode,
                candidate_k=candidate_candidate_k,
                dual_candidate_k=candidate_dual_candidate_k,
                hybrid_alpha=candidate_hybrid_alpha,
                reranker_enabled=bool(candidate_reranker),
            ),
            k=k,
            reranker_candidate_k=eval_bundle.reranker_candidate_k,
            reranker_strategy=eval_bundle.reranker_strategy,
            max_queries=max_queries,
            min_delta_ndcg=min_delta_ndcg,
            min_delta_map=min_delta_map,
            min_delta_mrr=min_delta_mrr,
            max_regression_precision=max_regression_precision,
            max_regression_recall=max_regression_recall,
        )
        if json_out is not None:
            json_out.write_text(json.dumps(eval_compare_result_to_json(result)), encoding="utf-8")

        baseline_line, candidate_line, delta_line, gate_line = format_eval_compare_result(result)
        click.echo(baseline_line)
        click.echo(candidate_line)
        click.echo(delta_line)
        if result.gate.passed:
            click.echo(gate_line)
            return
        click.echo(gate_line, err=True)
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"[ERROR] Error comparing eval runs: {e}", err=True)
        raise SystemExit(2)
