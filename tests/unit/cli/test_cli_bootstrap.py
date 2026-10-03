from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from local_rag_backend.cli_commands.docs import docs_bootstrap as bootstrap_cmd_module
from local_rag_backend.scripts.sample_data_ingestion import _build_bootstrap_mutation_intent
from local_rag_backend.settings import get_settings

settings = get_settings()


def test_bootstrap_cli_runs_sample_data_ingestion(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def _fake_run_cli_mutation(operation, **kwargs):
        calls["mutation_kwargs"] = kwargs
        return operation()

    def _fake_run_sample_data_ingestion(*, settings_obj=settings, **kwargs):
        calls["settings_obj"] = settings_obj
        calls["ingestion_kwargs"] = kwargs
        return 1

    monkeypatch.setattr(
        bootstrap_cmd_module,
        "run_cli_mutation",
        _fake_run_cli_mutation,
        raising=True,
    )
    monkeypatch.setattr(
        "local_rag_backend.scripts.sample_data_ingestion.run_sample_data_ingestion",
        _fake_run_sample_data_ingestion,
        raising=True,
    )

    result = CliRunner().invoke(bootstrap_cmd_module.bootstrap_cmd)
    assert result.exit_code == 0, result.output
    assert "Bootstrap completed successfully!" in result.output
    assert calls["settings_obj"] is settings
    assert calls["ingestion_kwargs"] == {}
    assert calls["mutation_kwargs"] == {
        "use_lock": False,
        "ensure_schema": False,
        "invalidate_shared": False,
    }


def test_bootstrap_cli_reports_ingestion_failure(monkeypatch) -> None:
    def _boom(operation, **kwargs):
        _ = (operation, kwargs)
        raise RuntimeError("sample ingestion failed")

    monkeypatch.setattr(bootstrap_cmd_module, "run_cli_mutation", _boom, raising=True)

    result = CliRunner().invoke(bootstrap_cmd_module.bootstrap_cmd)

    assert result.exit_code == 1
    assert "[ERROR] Error bootstrapping: sample ingestion failed" in result.output


def test_bootstrap_skips_blank_chunks_and_empty_rows(tmp_path) -> None:
    csv_file = tmp_path / "faq.csv"
    csv_file.write_text("Q;A\nQ;A        X\n;\n", encoding="utf-8")
    settings_obj = SimpleNamespace(
        csv_has_header=True, ingest_chunk_chars=4, ingest_chunk_overlap=0
    )
    repo = SimpleNamespace(
        get_tombstoned_external_ids=lambda _ids: set(),
        list_ids_by_external_id_prefix=lambda _prefix: [],
    )
    ports = SimpleNamespace(doc_repo_factory=lambda: repo)

    intent = _build_bootstrap_mutation_intent(
        csv_path_obj=csv_file, settings_obj=settings_obj, ports=ports
    )

    assert [item.content for item in intent.upserts] == ["Q\n\nA", "X"]
    assert [item.metadata["chunk_index"] for item in intent.upserts] == [0, 3]
    assert [item.metadata["chunk_start_char"] for item in intent.upserts] == [0, 12]
    assert [item.metadata["chunk_end_char"] for item in intent.upserts] == [4, 13]


def test_bootstrap_all_blank_rows_produce_noop_intent(tmp_path) -> None:
    csv_file = tmp_path / "faq.csv"
    csv_file.write_text("Q;A\n;\n", encoding="utf-8")
    settings_obj = SimpleNamespace(
        csv_has_header=True, ingest_chunk_chars=4, ingest_chunk_overlap=0
    )
    repo = SimpleNamespace(
        get_tombstoned_external_ids=lambda _ids: set(),
        list_ids_by_external_id_prefix=lambda _prefix: [],
    )
    ports = SimpleNamespace(doc_repo_factory=lambda: repo)

    intent = _build_bootstrap_mutation_intent(
        csv_path_obj=csv_file, settings_obj=settings_obj, ports=ports
    )

    assert intent.upserts == ()
    assert intent.delete_ids == ()
