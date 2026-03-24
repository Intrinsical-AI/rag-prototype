"""Application-layer orchestration for offline retrieval evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    retrieval_result_from_pairs,
)
from local_rag_backend.core.ports import (
    EvalDatasetDocInput,
    EvalRetrieverFactoryPort,
    EvalRetrieverPort,
    EvalStoragePort,
)
from local_rag_backend.core.services.evaluation import (
    EvalCompareResult,
    EvalDataset,
    EvalResult,
    compare_eval_results,
    run_retrieval_eval as run_retrieval_eval_core,
)
from local_rag_backend.core.services.types import (
    EvalBatchResult,
    EvalBatchSpec,
    EvalCompareConfig,
    EvalRetrievalConfig,
    EvalRetrievalMode,
)


@dataclass(frozen=True)
class PreparedEvalWorkspace:
    dataset: EvalDataset
    eval_storage_port: EvalStoragePort
    eval_retriever_factory_port: EvalRetrieverFactoryPort
    prepared_retriever_workspace: Any | None = None


def _dataset_doc_inputs(dataset: EvalDataset) -> tuple[EvalDatasetDocInput, ...]:
    return tuple(
        EvalDatasetDocInput(
            external_id=d.external_id,
            content=d.content,
            source_id=d.source_id,
            metadata={"dataset_id": dataset.dataset_id},
        )
        for d in dataset.docs
    )


def _resolve_prepared_retriever_workspace(
    *,
    eval_retriever_factory_port: EvalRetrieverFactoryPort,
    eval_storage_port: EvalStoragePort,
) -> Any | None:
    prepare_workspace = getattr(eval_retriever_factory_port, "prepare_workspace", None)
    if not callable(prepare_workspace):
        return None
    return prepare_workspace(storage=eval_storage_port)


def _external_ids_from_retrieval(retrieval: RetrievalResult) -> list[str]:
    return [
        str(external_id)
        for external_id in (
            getattr(document, "external_id", None) for document in retrieval.documents
        )
        if external_id is not None and str(external_id).strip()
    ]


def _rankings_lookup(rankings: dict[str, list[str]]) -> Callable[[str, int], Sequence[str]]:
    def retrieve_external_ids(query: str, _top_k: int) -> list[str]:
        return rankings.get(query, [])

    return retrieve_external_ids


def _coerce_eval_retrieval_mode(retrieval_mode: str) -> EvalRetrievalMode:
    normalized = str(retrieval_mode)
    if normalized not in {"sparse", "dense", "dual", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
    return cast("EvalRetrievalMode", normalized)


def _run_eval_with_retriever(
    *,
    dataset: EvalDataset,
    retriever: EvalRetrieverPort,
    retrieval_mode: EvalRetrievalMode,
    k: int,
    reranker_enabled: bool,
    candidate_k: int | None,
    dual_candidate_k: int | None,
    max_queries: int | None,
    run_out: Path | None,
) -> EvalResult:
    def _retrieve_external_ids(query: str, top_k: int) -> list[str]:
        request = RetrievalRequest(
            query=query,
            top_k=top_k,
            mode=retrieval_mode,
            candidate_k=(int(candidate_k) if candidate_k is not None else None),
            dual_candidate_k=(int(dual_candidate_k) if dual_candidate_k is not None else None),
        )
        retrieval = _coerce_retrieval_result(retriever.retrieve(request), request=request)
        return _external_ids_from_retrieval(retrieval)

    return run_retrieval_eval_core(
        dataset=dataset,
        retrieve_external_ids=_retrieve_external_ids,
        retrieval_mode=retrieval_mode,
        k=k,
        reranker_enabled=reranker_enabled,
        max_queries=max_queries,
        run_out=run_out,
    )


def prepare_eval_workspace(
    *,
    dataset: EvalDataset,
    eval_storage_port: EvalStoragePort,
    eval_retriever_factory_port: EvalRetrieverFactoryPort,
) -> PreparedEvalWorkspace:
    eval_storage_port.upsert_dataset_docs(
        dataset_id=dataset.dataset_id, docs=_dataset_doc_inputs(dataset)
    )
    return PreparedEvalWorkspace(
        dataset=dataset,
        eval_storage_port=eval_storage_port,
        eval_retriever_factory_port=eval_retriever_factory_port,
        prepared_retriever_workspace=_resolve_prepared_retriever_workspace(
            eval_retriever_factory_port=eval_retriever_factory_port,
            eval_storage_port=eval_storage_port,
        ),
    )


def _coerce_retrieval_result(
    raw_result: Any,
    *,
    request: RetrievalRequest,
) -> RetrievalResult:
    if isinstance(raw_result, RetrievalResult):
        return raw_result
    if (
        isinstance(raw_result, tuple)
        and len(raw_result) == 2
        and isinstance(raw_result[0], (list, tuple))
        and isinstance(raw_result[1], (list, tuple))
    ):
        docs, scores = raw_result
        return retrieval_result_from_pairs(
            docs=docs,
            scores=scores,
            mode_used=request.mode,
            backend_used="legacy_eval",
        )
    raise RuntimeError(f"Unsupported eval retriever response type: {type(raw_result)!r}")


def _validate_eval_options(
    *,
    retrieval_mode: str,
    candidate_k: int | None,
    dual_candidate_k: int | None,
    hybrid_alpha: float | None,
) -> None:
    if candidate_k is not None and retrieval_mode != "dense":
        raise ValueError("--candidate-k is supported only with retrieval_mode=dense")
    if dual_candidate_k is not None and retrieval_mode != "dual":
        raise ValueError("--dual-candidate-k is supported only with retrieval_mode=dual")
    if hybrid_alpha is not None and retrieval_mode != "hybrid":
        raise ValueError("--hybrid-alpha is supported only with retrieval_mode=hybrid")
    if candidate_k is not None and int(candidate_k) <= 0:
        raise ValueError("candidate_k must be positive")
    if dual_candidate_k is not None and int(dual_candidate_k) <= 0:
        raise ValueError("dual_candidate_k must be positive")
    if hybrid_alpha is not None and not 0.0 <= float(hybrid_alpha) <= 1.0:
        raise ValueError("hybrid_alpha must be between 0.0 and 1.0")


def run_retrieval_eval(
    *,
    dataset: EvalDataset,
    eval_storage_port: EvalStoragePort,
    eval_retriever_factory_port: EvalRetrieverFactoryPort,
    retrieval_mode: str = "sparse",
    k: int = 3,
    candidate_k: int | None = None,
    reranker_enabled: bool = False,
    dual_candidate_k: int | None = None,
    hybrid_alpha: float | None = None,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
    run_out: Path | None = None,
) -> EvalResult:
    if retrieval_mode not in {"sparse", "dense", "dual", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
    if k <= 0:
        raise ValueError("k must be positive")
    _validate_eval_options(
        retrieval_mode=str(retrieval_mode),
        candidate_k=candidate_k,
        dual_candidate_k=dual_candidate_k,
        hybrid_alpha=hybrid_alpha,
    )
    normalized_retrieval_mode = _coerce_eval_retrieval_mode(retrieval_mode)
    workspace = prepare_eval_workspace(
        dataset=dataset,
        eval_storage_port=eval_storage_port,
        eval_retriever_factory_port=eval_retriever_factory_port,
    )
    return run_prepared_retrieval_eval(
        workspace=workspace,
        config=EvalRetrievalConfig(
            retrieval_mode=normalized_retrieval_mode,
            k=int(k),
            candidate_k=(int(candidate_k) if candidate_k is not None else None),
            dual_candidate_k=(int(dual_candidate_k) if dual_candidate_k is not None else None),
            hybrid_alpha=(float(hybrid_alpha) if hybrid_alpha is not None else None),
            reranker_enabled=bool(reranker_enabled),
        ),
        reranker_candidate_k=reranker_candidate_k,
        reranker_strategy=reranker_strategy,
        max_queries=max_queries,
        run_out=run_out,
    )


def run_prepared_retrieval_eval(
    *,
    workspace: PreparedEvalWorkspace,
    config: EvalRetrievalConfig,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
    run_out: Path | None = None,
) -> EvalResult:
    _validate_eval_options(
        retrieval_mode=str(config.retrieval_mode),
        candidate_k=config.candidate_k,
        dual_candidate_k=config.dual_candidate_k,
        hybrid_alpha=config.hybrid_alpha,
    )
    prepared = workspace.prepared_retriever_workspace
    if prepared is not None and hasattr(prepared, "build_retriever"):
        retriever = prepared.build_retriever(
            config=config,
            reranker_candidate_k=reranker_candidate_k,
            reranker_strategy=reranker_strategy,
        )
    else:
        retriever = workspace.eval_retriever_factory_port.build_retriever(
            storage=workspace.eval_storage_port,
            config=config,
            reranker_candidate_k=reranker_candidate_k,
            reranker_strategy=reranker_strategy,
        )
    return _run_eval_with_retriever(
        dataset=workspace.dataset,
        retriever=retriever,
        retrieval_mode=config.retrieval_mode,
        k=int(config.k),
        reranker_enabled=bool(config.reranker_enabled),
        candidate_k=config.candidate_k,
        dual_candidate_k=config.dual_candidate_k,
        max_queries=max_queries,
        run_out=run_out,
    )


def _effective_hybrid_alpha(*, workspace: PreparedEvalWorkspace, spec: EvalBatchSpec) -> float:
    if spec.hybrid_alpha is not None:
        return float(spec.hybrid_alpha)
    return float(workspace.eval_storage_port.get_eval_settings().hybrid_retrieval_alpha)


def _run_exact_hybrid_alpha_group(
    *,
    workspace: PreparedEvalWorkspace,
    specs: tuple[EvalBatchSpec, ...],
    reranker_candidate_k: int,
    reranker_strategy: str,
    max_queries: int | None,
) -> tuple[EvalBatchResult, ...]:
    prepared = workspace.prepared_retriever_workspace
    retrieve_group = getattr(prepared, "retrieve_hybrid_alpha_group", None)
    if not callable(retrieve_group):
        return tuple(
            EvalBatchResult(
                name=spec.name,
                result=run_prepared_retrieval_eval(
                    workspace=workspace,
                    config=EvalRetrievalConfig(
                        retrieval_mode=spec.retrieval_mode,
                        k=spec.k,
                        candidate_k=spec.candidate_k,
                        dual_candidate_k=spec.dual_candidate_k,
                        hybrid_alpha=spec.hybrid_alpha,
                        reranker_enabled=spec.reranker_enabled,
                    ),
                    reranker_candidate_k=reranker_candidate_k,
                    reranker_strategy=reranker_strategy,
                    max_queries=max_queries,
                    run_out=(Path(spec.run_out) if spec.run_out is not None else None),
                ),
                json_out=spec.json_out,
                run_out=spec.run_out,
            )
            for spec in specs
        )

    k = int(specs[0].k)
    if any(spec.reranker_enabled for spec in specs):
        raise AssertionError(
            "Exact hybrid alpha sweep is supported only when reranker is disabled."
        )
    if any(int(spec.k) != k for spec in specs):
        raise AssertionError("Exact hybrid alpha sweep requires all grouped specs to share k.")

    alpha_by_name = {
        spec.name: _effective_hybrid_alpha(workspace=workspace, spec=spec) for spec in specs
    }
    ranking_by_name: dict[str, dict[str, list[str]]] = {spec.name: {} for spec in specs}
    queries = list(workspace.dataset.queries)
    if max_queries is not None:
        queries = queries[: max(0, int(max_queries))]

    alphas = tuple(dict.fromkeys(alpha_by_name.values()))
    for query in queries:
        hybrid_results = retrieve_group(query=query.query, top_k=k, alphas=alphas)
        for spec in specs:
            retrieval = hybrid_results[alpha_by_name[spec.name]]
            ranking_by_name[spec.name][query.query] = _external_ids_from_retrieval(retrieval)

    out: list[EvalBatchResult] = []
    for spec in specs:
        run_out_path = Path(spec.run_out) if spec.run_out is not None else None
        rankings = ranking_by_name[spec.name]

        result = run_retrieval_eval_core(
            dataset=workspace.dataset,
            retrieve_external_ids=_rankings_lookup(rankings),
            retrieval_mode="hybrid",
            k=k,
            reranker_enabled=False,
            max_queries=max_queries,
            run_out=run_out_path,
        )
        out.append(
            EvalBatchResult(
                name=spec.name,
                result=result,
                json_out=spec.json_out,
                run_out=spec.run_out,
            )
        )
    return tuple(out)


def run_retrieval_eval_batch(
    *,
    dataset: EvalDataset,
    eval_storage_port: EvalStoragePort,
    eval_retriever_factory_port: EvalRetrieverFactoryPort,
    specs: tuple[EvalBatchSpec, ...],
    fresh_workspace: bool = False,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
) -> tuple[EvalBatchResult, ...]:
    if not specs:
        return ()
    names = [spec.name.strip() for spec in specs]
    if any(not name for name in names):
        raise ValueError("Eval batch spec name must not be blank.")
    if len(set(names)) != len(names):
        raise ValueError("Eval batch spec names must be unique.")

    if fresh_workspace:
        return tuple(
            EvalBatchResult(
                name=spec.name,
                result=run_prepared_retrieval_eval(
                    workspace=prepare_eval_workspace(
                        dataset=dataset,
                        eval_storage_port=eval_storage_port,
                        eval_retriever_factory_port=eval_retriever_factory_port,
                    ),
                    config=EvalRetrievalConfig(
                        retrieval_mode=spec.retrieval_mode,
                        k=spec.k,
                        candidate_k=spec.candidate_k,
                        dual_candidate_k=spec.dual_candidate_k,
                        hybrid_alpha=spec.hybrid_alpha,
                        reranker_enabled=spec.reranker_enabled,
                    ),
                    reranker_candidate_k=reranker_candidate_k,
                    reranker_strategy=reranker_strategy,
                    max_queries=max_queries,
                    run_out=(Path(spec.run_out) if spec.run_out is not None else None),
                ),
                json_out=spec.json_out,
                run_out=spec.run_out,
            )
            for spec in specs
        )

    workspace = prepare_eval_workspace(
        dataset=dataset,
        eval_storage_port=eval_storage_port,
        eval_retriever_factory_port=eval_retriever_factory_port,
    )
    grouped_hybrid: dict[int, list[EvalBatchSpec]] = {}
    exact_names: set[str] = set()
    for spec in specs:
        if spec.retrieval_mode == "hybrid" and not spec.reranker_enabled:
            grouped_hybrid.setdefault(int(spec.k), []).append(spec)
            exact_names.add(spec.name)

    results: list[EvalBatchResult] = []
    for spec in specs:
        if spec.name in exact_names:
            continue
        result = run_prepared_retrieval_eval(
            workspace=workspace,
            config=EvalRetrievalConfig(
                retrieval_mode=spec.retrieval_mode,
                k=spec.k,
                candidate_k=spec.candidate_k,
                dual_candidate_k=spec.dual_candidate_k,
                hybrid_alpha=spec.hybrid_alpha,
                reranker_enabled=spec.reranker_enabled,
            ),
            reranker_candidate_k=reranker_candidate_k,
            reranker_strategy=reranker_strategy,
            max_queries=max_queries,
            run_out=(Path(spec.run_out) if spec.run_out is not None else None),
        )
        results.append(
            EvalBatchResult(
                name=spec.name,
                result=result,
                json_out=spec.json_out,
                run_out=spec.run_out,
            )
        )

    for k in sorted(grouped_hybrid):
        results.extend(
            _run_exact_hybrid_alpha_group(
                workspace=workspace,
                specs=tuple(grouped_hybrid[k]),
                reranker_candidate_k=reranker_candidate_k,
                reranker_strategy=reranker_strategy,
                max_queries=max_queries,
            )
        )

    order = {spec.name: index for index, spec in enumerate(specs)}
    return tuple(sorted(results, key=lambda item: order[item.name]))


def compare_retrieval_eval(
    *,
    dataset: EvalDataset,
    eval_storage_port: EvalStoragePort,
    eval_retriever_factory_port: EvalRetrieverFactoryPort,
    baseline: EvalCompareConfig,
    candidate: EvalCompareConfig,
    k: int = 3,
    reranker_candidate_k: int = 20,
    reranker_strategy: str = "overlap_v1",
    max_queries: int | None = None,
    min_delta_ndcg: float = 0.0,
    min_delta_map: float = 0.0,
    min_delta_mrr: float = 0.0,
    max_regression_precision: float = 0.0,
    max_regression_recall: float = 0.0,
) -> EvalCompareResult:
    baseline_result = run_retrieval_eval(
        dataset=dataset,
        eval_storage_port=eval_storage_port,
        eval_retriever_factory_port=eval_retriever_factory_port,
        retrieval_mode=baseline.retrieval_mode,
        k=k,
        candidate_k=baseline.candidate_k,
        reranker_enabled=baseline.reranker_enabled,
        dual_candidate_k=baseline.dual_candidate_k,
        hybrid_alpha=baseline.hybrid_alpha,
        reranker_candidate_k=reranker_candidate_k,
        reranker_strategy=reranker_strategy,
        max_queries=max_queries,
    )
    candidate_result = run_retrieval_eval(
        dataset=dataset,
        eval_storage_port=eval_storage_port,
        eval_retriever_factory_port=eval_retriever_factory_port,
        retrieval_mode=candidate.retrieval_mode,
        k=k,
        candidate_k=candidate.candidate_k,
        reranker_enabled=candidate.reranker_enabled,
        dual_candidate_k=candidate.dual_candidate_k,
        hybrid_alpha=candidate.hybrid_alpha,
        reranker_candidate_k=reranker_candidate_k,
        reranker_strategy=reranker_strategy,
        max_queries=max_queries,
    )
    return compare_eval_results(
        baseline=baseline_result,
        candidate=candidate_result,
        min_delta_ndcg=min_delta_ndcg,
        min_delta_map=min_delta_map,
        min_delta_mrr=min_delta_mrr,
        max_regression_precision=max_regression_precision,
        max_regression_recall=max_regression_recall,
    )
