from __future__ import annotations

from types import SimpleNamespace

from local_rag_backend.core.use_cases import health


def test_check_sql_counts_marks_documents_failure_without_failing_history() -> None:
    checks: dict[str, object] = {}

    class _Diagnostics:
        def get_documents_count(self):
            raise RuntimeError("docs unavailable")

        def get_history_count(self):
            return 5

    ready, docs_count = health.check_sql_counts(checks=checks, diagnostics=_Diagnostics())

    assert ready is False
    assert docs_count is None
    assert checks["documents"] == "failed: unavailable"
    assert checks["history"] == {"count": 5}


def test_check_retrieval_index_skips_non_dense_modes() -> None:
    checks: dict[str, object] = {}

    class _Diagnostics:
        def get_retrieval_index_stats(self, **kwargs):
            raise AssertionError("should not be called")

    assert (
        health.check_retrieval_index(
            checks=checks,
            docs_count=1,
            settings_obj=SimpleNamespace(retrieval_mode="sparse"),
            diagnostics=_Diagnostics(),
        )
        is True
    )


def test_check_retrieval_index_handles_count_mismatch_and_id_drift_errors() -> None:
    settings_obj = SimpleNamespace(
        retrieval_mode="dense",
        index_path="index.faiss",
        id_map_path="id_map.json",
        vector_backend="numpy",
    )

    class _CountMismatchDiagnostics:
        def get_retrieval_index_stats(self, **kwargs):
            _ = kwargs
            return {
                "status": "ok",
                "vectors": 1,
                "id_map_len": 1,
                "index_path": "index.faiss",
                "id_map_path": "id_map.json",
            }

    checks: dict[str, object] = {}
    assert (
        health.check_retrieval_index(
            checks=checks,
            docs_count=2,
            settings_obj=settings_obj,
            diagnostics=_CountMismatchDiagnostics(),
        )
        is False
    )
    assert "documents=2, vectors=1" in str(checks["retrieval_index"])

    class _IdDriftDiagnostics:
        def get_retrieval_index_stats(self, **kwargs):
            _ = kwargs
            return {
                "status": "ok",
                "vectors": 2,
                "id_map_len": 2,
                "index_path": "index.faiss",
                "id_map_path": "id_map.json",
            }

        def get_document_ids(self):
            return ["doc-1", "doc-2"]

        def get_index_ids(self, **kwargs):
            _ = kwargs
            raise RuntimeError("cannot read id map")

    checks = {}
    assert (
        health.check_retrieval_index(
            checks=checks,
            docs_count=2,
            settings_obj=settings_obj,
            diagnostics=_IdDriftDiagnostics(),
        )
        is False
    )
    assert checks["retrieval_index_drift"] == "failed: unavailable"
    assert str(checks["retrieval_index"]).startswith(
        "failed: unable to verify retrieval index drift"
    )


def test_check_mutation_journal_blocks_readiness_on_pending_or_unreadable_records() -> None:
    class _PendingDiagnostics:
        def get_incomplete_mutation_records_count(self, *, coordination_dir):
            assert coordination_dir == "/coord"
            return 2

    checks = {}
    assert (
        health.check_mutation_journal(
            checks=checks,
            settings_obj=SimpleNamespace(
                get_coordination_dir=lambda: "/coord",
            ),
            diagnostics=_PendingDiagnostics(),
        )
        is False
    )
    assert checks["mutation_journal"] == {"status": "failed", "incomplete_records": 2}

    class _FailingDiagnostics:
        def get_incomplete_mutation_records_count(self, *, coordination_dir):
            _ = coordination_dir
            raise RuntimeError("journal unavailable")

    checks = {}
    assert (
        health.check_mutation_journal(
            checks=checks,
            settings_obj=SimpleNamespace(
                get_coordination_dir=lambda: "/coord",
            ),
            diagnostics=_FailingDiagnostics(),
        )
        is False
    )
    assert checks["mutation_journal"] == {"status": "failed"}
