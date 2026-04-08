# tests/unit/app/services/test_evaluation.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from local_rag_backend.composition import adapters as adapters_module
from local_rag_backend.composition.adapters import (
    build_eval_retriever_factory_port,
    build_eval_storage_port,
)
from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    retrieval_result_from_pairs,
)
from local_rag_backend.core.ports import EvalDatasetDocInput
from local_rag_backend.core.services.evaluation import (
    EvalDataset,
    EvalDoc,
    EvalQuery,
    load_eval_dataset,
    run_retrieval_eval as run_core_eval,
)
from local_rag_backend.core.services.types import (
    EvalBatchSpec,
    EvalCompareConfig,
    EvalRetrievalConfig,
)
from local_rag_backend.core.use_cases.evaluation import (
    compare_retrieval_eval,
    run_retrieval_eval,
    run_retrieval_eval_batch,
)
from local_rag_backend.settings import settings


class DummyEmbedder:
    dim = 2

    def embed(self, texts):
        out = []
        for text in texts:
            normalized = text.lower()
            if "auth" in normalized:
                out.append([1.0, 0.0])
            elif "sql" in normalized:
                out.append([0.0, 1.0])
            else:
                out.append([0.5, 0.5])
        return out


def _coerce_retrieval_result(raw_result: Any, *, request: RetrievalRequest) -> RetrievalResult:
    if isinstance(raw_result, RetrievalResult):
        return raw_result
    docs, scores = raw_result
    return retrieval_result_from_pairs(
        docs=docs,
        scores=scores,
        mode_used=request.mode,
        backend_used="legacy_test",
    )


def _eval_settings():
    return settings.model_copy(
        update={
            "persistence_backend": "local_split",
            "search_backend": "local_split",
            "vector_backend": "numpy",
            "openai_api_key": None,
        }
    )


def _mode_dataset() -> EvalDataset:
    return EvalDataset(
        dataset_id="multi-mode",
        schema_version=1,
        docs=(
            EvalDoc(external_id="doc-auth", content="auth auth guard"),
            EvalDoc(external_id="doc-sql", content="sql sql helper"),
        ),
        queries=(
            EvalQuery(query="auth", relevant_external_ids=("doc-auth",)),
            EvalQuery(query="sql", relevant_external_ids=("doc-sql",)),
        ),
    )


def test_run_retrieval_eval_app_service_passes_on_default_repo_dataset() -> None:
    ds = load_eval_dataset()
    cfg = _eval_settings()
    res = run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(settings_obj=cfg),
        eval_retriever_factory_port=build_eval_retriever_factory_port(
            st_embedder_factory=lambda _model_name: DummyEmbedder(),
        ),
        retrieval_mode="sparse",
        reranker_enabled=True,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
    )
    assert res.queries > 0
    assert 0.0 <= res.ndcg_at_k <= 1.0
    assert 0.0 <= res.map_at_k <= 1.0
    assert 0.0 <= res.mrr_at_k <= 1.0
    assert 0.0 <= res.precision_at_k <= 1.0
    assert 0.0 <= res.recall_at_k <= 1.0


def test_run_retrieval_eval_app_service_matches_core_semantics() -> None:
    ds = load_eval_dataset()
    cfg = _eval_settings()
    storage = build_eval_storage_port(settings_obj=cfg)
    storage.upsert_dataset_docs(
        dataset_id=ds.dataset_id,
        docs=tuple(
            EvalDatasetDocInput(
                external_id=d.external_id,
                content=d.content,
                source_id=d.source_id,
                metadata={"dataset_id": ds.dataset_id},
            )
            for d in ds.docs
        ),
    )
    retriever = build_eval_retriever_factory_port(
        st_embedder_factory=lambda _model_name: DummyEmbedder(),
    ).build_retriever(
        storage=storage,
        config=EvalRetrievalConfig(
            retrieval_mode="sparse",
            k=3,
            reranker_enabled=True,
        ),
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
    )

    def _retrieve_external_ids(query: str, top_k: int) -> list[str]:
        request = RetrievalRequest(query=query, top_k=top_k, mode="sparse")
        retrieval = _coerce_retrieval_result(retriever.retrieve(request), request=request)
        return [
            str(external_id)
            for external_id in (
                getattr(document, "external_id", None) for document in retrieval.documents
            )
            if external_id is not None and str(external_id).strip()
        ]

    core_res = run_core_eval(
        dataset=ds,
        retrieve_external_ids=_retrieve_external_ids,
        retrieval_mode="sparse",
        k=3,
        reranker_enabled=True,
    )
    app_res = run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(settings_obj=cfg),
        eval_retriever_factory_port=build_eval_retriever_factory_port(
            st_embedder_factory=lambda _model_name: DummyEmbedder(),
        ),
        retrieval_mode="sparse",
        k=3,
        reranker_enabled=True,
        reranker_candidate_k=20,
        reranker_strategy="overlap_v1",
    )

    assert app_res == core_res


def test_build_eval_storage_port_uses_isolated_local_paths(tmp_path: Path) -> None:
    base = settings.model_copy(
        update={
            "data_dir": tmp_path / "main-data",
            "index_path": str(tmp_path / "main.index"),
            "id_map_path": str(tmp_path / "main_id_map.json"),
            "sqlite_url": f"sqlite:///{tmp_path / 'main.db'}",
            "persistence_backend": "elasticsearch",
            "search_backend": "elasticsearch",
        }
    )

    storage = build_eval_storage_port(settings_obj=base)
    eval_settings = storage.get_eval_settings()

    assert eval_settings.persistence_backend == "local_split"
    assert eval_settings.search_backend == "local_split"
    assert Path(eval_settings.index_path) != Path(base.index_path)
    assert Path(eval_settings.id_map_path) != Path(base.id_map_path)
    assert eval_settings.sqlite_url != base.sqlite_url


def test_build_eval_storage_port_uses_deterministic_dataset_workspace(tmp_path: Path) -> None:
    cfg = settings.model_copy(
        update={
            "data_dir": tmp_path / "main-data",
            "vector_backend": "numpy",
            "openai_api_key": None,
        }
    )
    ds = _mode_dataset()

    storage_a = build_eval_storage_port(settings_obj=cfg)
    storage_a.upsert_dataset_docs(
        dataset_id=ds.dataset_id,
        docs=tuple(
            EvalDatasetDocInput(
                external_id=d.external_id,
                content=d.content,
                source_id=d.source_id,
                metadata={"dataset_id": ds.dataset_id},
            )
            for d in ds.docs
        ),
    )
    settings_a = storage_a.get_eval_settings()

    storage_b = build_eval_storage_port(settings_obj=cfg)
    storage_b.upsert_dataset_docs(
        dataset_id=ds.dataset_id,
        docs=tuple(
            EvalDatasetDocInput(
                external_id=d.external_id,
                content=d.content,
                source_id=d.source_id,
                metadata={"dataset_id": ds.dataset_id},
            )
            for d in ds.docs
        ),
    )
    settings_b = storage_b.get_eval_settings()

    assert Path(settings_a.data_dir) == Path(settings_b.data_dir)
    assert Path(settings_a.data_dir).parent.name == "_eval_workspaces"
    assert Path(settings_a.index_path).parent == Path(settings_a.data_dir)
    assert Path(settings_a.id_map_path).parent == Path(settings_a.data_dir)


@pytest.mark.parametrize(
    ("retrieval_mode", "extra_kwargs"),
    [
        ("dense", {"candidate_k": 1}),
        ("dual", {"dual_candidate_k": 2}),
        ("hybrid", {"hybrid_alpha": 0.5}),
    ],
)
def test_run_retrieval_eval_app_service_supports_multi_mode_runtime(
    retrieval_mode: str,
    extra_kwargs: dict[str, Any],
) -> None:
    ds = _mode_dataset()
    cfg = _eval_settings()

    res = run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(settings_obj=cfg),
        eval_retriever_factory_port=build_eval_retriever_factory_port(
            st_embedder_factory=lambda _model_name: DummyEmbedder(),
        ),
        retrieval_mode=retrieval_mode,
        k=1,
        reranker_enabled=False,
        **extra_kwargs,
    )

    assert res.retrieval_mode == retrieval_mode
    assert res.ndcg_at_k == pytest.approx(1.0)
    assert res.map_at_k == pytest.approx(1.0)
    assert res.mrr_at_k == pytest.approx(1.0)
    assert res.precision_at_k == pytest.approx(1.0)
    assert res.recall_at_k == pytest.approx(1.0)


def test_compare_retrieval_eval_uses_same_dataset_and_runtime_for_baseline_and_candidate() -> None:
    ds = _mode_dataset()
    cfg = _eval_settings()
    storage = build_eval_storage_port(settings_obj=cfg)

    result = compare_retrieval_eval(
        dataset=ds,
        eval_storage_port=storage,
        eval_retriever_factory_port=build_eval_retriever_factory_port(
            st_embedder_factory=lambda _model_name: DummyEmbedder(),
        ),
        baseline=EvalCompareConfig(retrieval_mode="sparse"),
        candidate=EvalCompareConfig(retrieval_mode="dual", dual_candidate_k=2),
        k=1,
        min_delta_ndcg=0.0,
        min_delta_map=0.0,
        min_delta_mrr=0.0,
        max_regression_precision=0.0,
        max_regression_recall=0.0,
    )

    assert result.dataset_id == ds.dataset_id
    assert result.baseline.retrieval_mode == "sparse"
    assert result.candidate.retrieval_mode == "dual"
    assert result.baseline.queries == result.candidate.queries == len(ds.queries)


def test_compare_retrieval_eval_keeps_prepared_workspaces_independent() -> None:
    ds = _mode_dataset()
    cfg = _eval_settings()
    storage = build_eval_storage_port(settings_obj=cfg)
    factory_calls = {"count": 0}

    def _st_embedder(_model_name: str):
        factory_calls["count"] += 1
        return DummyEmbedder()

    result = compare_retrieval_eval(
        dataset=ds,
        eval_storage_port=storage,
        eval_retriever_factory_port=build_eval_retriever_factory_port(
            st_embedder_factory=_st_embedder,
        ),
        baseline=EvalCompareConfig(retrieval_mode="dense", candidate_k=1),
        candidate=EvalCompareConfig(retrieval_mode="hybrid", hybrid_alpha=0.5),
        k=1,
    )

    assert result.baseline.retrieval_mode == "dense"
    assert result.candidate.retrieval_mode == "hybrid"
    assert factory_calls["count"] == 2


def test_run_retrieval_eval_batch_matches_individual_runs_for_exact_hybrid_sweep(
    tmp_path: Path,
) -> None:
    ds = _mode_dataset()
    cfg = _eval_settings()
    storage = build_eval_storage_port(settings_obj=cfg)
    factory = build_eval_retriever_factory_port(
        st_embedder_factory=lambda _model_name: DummyEmbedder(),
    )
    specs = (
        EvalBatchSpec(
            name="hybrid-02",
            retrieval_mode="hybrid",
            k=1,
            hybrid_alpha=0.2,
            reranker_enabled=False,
            json_out=str(tmp_path / "hybrid-02.json"),
            run_out=str(tmp_path / "hybrid-02.jsonl"),
        ),
        EvalBatchSpec(
            name="hybrid-08",
            retrieval_mode="hybrid",
            k=1,
            hybrid_alpha=0.8,
            reranker_enabled=False,
            json_out=str(tmp_path / "hybrid-08.json"),
            run_out=str(tmp_path / "hybrid-08.jsonl"),
        ),
    )

    batch_results = run_retrieval_eval_batch(
        dataset=ds,
        eval_storage_port=storage,
        eval_retriever_factory_port=factory,
        specs=specs,
    )
    single_02 = run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(settings_obj=cfg),
        eval_retriever_factory_port=factory,
        retrieval_mode="hybrid",
        k=1,
        hybrid_alpha=0.2,
    )
    single_08 = run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(settings_obj=cfg),
        eval_retriever_factory_port=factory,
        retrieval_mode="hybrid",
        k=1,
        hybrid_alpha=0.8,
    )

    assert [item.name for item in batch_results] == ["hybrid-02", "hybrid-08"]
    assert batch_results[0].result == single_02
    assert batch_results[1].result == single_08
    assert Path(specs[0].run_out or "").exists()
    assert Path(specs[1].run_out or "").exists()


def test_run_retrieval_eval_reuses_persisted_dense_eval_index(tmp_path: Path) -> None:
    ds = _mode_dataset()
    cfg = settings.model_copy(
        update={
            "data_dir": tmp_path / "eval-cache",
            "persistence_backend": "local_split",
            "search_backend": "local_split",
            "vector_backend": "numpy",
            "openai_api_key": None,
        }
    )
    batch_sizes: list[tuple[str, int]] = []

    class CountingEmbedder(DummyEmbedder):
        def __init__(self, label: str) -> None:
            self._label = label

        def embed(self, texts):
            batch_sizes.append((self._label, len(texts)))
            return super().embed(texts)

    run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(settings_obj=cfg),
        eval_retriever_factory_port=build_eval_retriever_factory_port(
            st_embedder_factory=lambda _model_name: CountingEmbedder("first"),
        ),
        retrieval_mode="dense",
        k=1,
        candidate_k=1,
        reranker_enabled=False,
    )
    run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(settings_obj=cfg),
        eval_retriever_factory_port=build_eval_retriever_factory_port(
            st_embedder_factory=lambda _model_name: CountingEmbedder("second"),
        ),
        retrieval_mode="dense",
        k=1,
        candidate_k=1,
        reranker_enabled=False,
    )

    first_sizes = [size for label, size in batch_sizes if label == "first"]
    second_sizes = [size for label, size in batch_sizes if label == "second"]

    assert len(ds.docs) in first_sizes
    assert len(ds.docs) not in second_sizes
    assert all(size == 1 for size in second_sizes)


def test_run_retrieval_eval_rebuilds_dense_eval_index_when_model_manifest_changes(
    tmp_path: Path,
) -> None:
    ds = _mode_dataset()
    base_cfg = {
        "data_dir": tmp_path / "eval-cache",
        "persistence_backend": "local_split",
        "search_backend": "local_split",
        "vector_backend": "numpy",
        "openai_api_key": None,
    }
    cfg_a = settings.model_copy(update={**base_cfg, "st_embedding_model": "model-a"})
    cfg_b = settings.model_copy(update={**base_cfg, "st_embedding_model": "model-b"})
    batch_sizes: list[tuple[str, int]] = []

    class CountingEmbedder(DummyEmbedder):
        def __init__(self, label: str) -> None:
            self._label = label

        def embed(self, texts):
            batch_sizes.append((self._label, len(texts)))
            return super().embed(texts)

    run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(settings_obj=cfg_a),
        eval_retriever_factory_port=build_eval_retriever_factory_port(
            st_embedder_factory=lambda _model_name: CountingEmbedder("first"),
        ),
        retrieval_mode="dense",
        k=1,
        candidate_k=1,
        reranker_enabled=False,
    )
    run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(settings_obj=cfg_b),
        eval_retriever_factory_port=build_eval_retriever_factory_port(
            st_embedder_factory=lambda _model_name: CountingEmbedder("second"),
        ),
        retrieval_mode="dense",
        k=1,
        candidate_k=1,
        reranker_enabled=False,
    )

    second_sizes = [size for label, size in batch_sizes if label == "second"]

    assert len(ds.docs) in second_sizes


def test_run_retrieval_eval_builds_dense_eval_index_in_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tuple(EvalDoc(external_id=f"doc-{idx}", content=f"topic {idx}") for idx in range(5))
    queries = (EvalQuery(query="topic 0", relevant_external_ids=("doc-0",)),)
    ds = EvalDataset(
        dataset_id="chunked-dense",
        schema_version=1,
        docs=docs,
        queries=queries,
    )
    cfg = settings.model_copy(
        update={
            "data_dir": tmp_path / "chunked-cache",
            "persistence_backend": "local_split",
            "search_backend": "local_split",
            "vector_backend": "numpy",
            "openai_api_key": None,
        }
    )
    batch_sizes: list[int] = []

    class CountingEmbedder(DummyEmbedder):
        def embed(self, texts):
            batch_sizes.append(len(texts))
            return super().embed(texts)

    monkeypatch.setattr(adapters_module, "EVAL_DENSE_REBUILD_BATCH_SIZE", 2)

    run_retrieval_eval(
        dataset=ds,
        eval_storage_port=build_eval_storage_port(settings_obj=cfg),
        eval_retriever_factory_port=build_eval_retriever_factory_port(
            st_embedder_factory=lambda _model_name: CountingEmbedder(),
        ),
        retrieval_mode="dense",
        k=1,
        candidate_k=1,
        reranker_enabled=False,
    )

    assert 2 in batch_sizes
    assert 1 in batch_sizes
