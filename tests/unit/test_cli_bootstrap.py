from __future__ import annotations

from click.testing import CliRunner

from local_rag_backend.cli_commands.docs import docs_bootstrap as bootstrap_cmd_module
from local_rag_backend.settings import settings


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
    assert calls["mutation_kwargs"] == {"use_lock": False, "ensure_schema": False}
