from __future__ import annotations

import json
from pathlib import Path

import click

from local_rag_backend.settings import settings


@click.command("eval")
@click.option(
    "--dataset",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a JSONL eval dataset. If omitted, uses the packaged rag_eval_v1 dataset.",
)
@click.option(
    "--retrieval-mode",
    type=click.Choice(["sparse"], case_sensitive=False),
    default="sparse",
    show_default=True,
)
@click.option("--k", type=int, default=3, show_default=True)
@click.option("--max-queries", type=int, default=None, help="Evaluate only the first N queries.")
@click.option("--reranker/--no-reranker", default=False, show_default=True)
@click.option("--fail-below-hit-rate", type=float, default=1.0, show_default=True)
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
    max_queries: int | None,
    reranker: bool,
    fail_below_hit_rate: float,
    fail_below_mrr: float,
    json_out: Path | None,
) -> None:
    """Offline retrieval evaluation (reproducible, dependency-free by default)."""
    from local_rag_backend.app.services.evaluation import run_retrieval_eval
    from local_rag_backend.core.services.evaluation import (
        eval_result_to_json,
        format_eval_result,
        load_eval_dataset,
    )

    ds = load_eval_dataset(dataset)
    res = run_retrieval_eval(
        dataset=ds,
        retrieval_mode=retrieval_mode,
        k=k,
        reranker_enabled=bool(reranker),
        reranker_candidate_k=settings.reranker_candidate_k,
        reranker_strategy=settings.reranker_strategy,
        max_queries=max_queries,
    )
    click.echo(format_eval_result(res))

    if json_out is not None:
        json_out.write_text(json.dumps(eval_result_to_json(res)), encoding="utf-8")

    if res.hit_rate < float(fail_below_hit_rate) or res.mrr < float(fail_below_mrr):
        click.echo(
            f"[ERROR] Eval regression: hit_rate={res.hit_rate:.3f} (min {fail_below_hit_rate}), "
            f"mrr={res.mrr:.3f} (min {fail_below_mrr})",
            err=True,
        )
        raise SystemExit(1)
