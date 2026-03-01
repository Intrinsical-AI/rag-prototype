from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from local_rag_backend.cli_commands import index as index_cmd_module
from local_rag_backend.settings import settings


def test_rebuild_index_cli_delegates_to_app_service(monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    calls: dict[str, object] = {}

    class _FakeContainer:
        settings_obj = settings

        def build_index_rebuild_bundle(self, **kwargs):
            calls["bundle_called"] = True
            calls["bundle_kwargs"] = kwargs
            return SimpleNamespace(ports="PORTS")

    def _fake_run_cli_mutation(operation, **kwargs):
        calls["mutation_kwargs"] = kwargs
        return operation()

    def _fake_rebuild_index_sync(*, settings_obj, ports):
        calls["settings_obj"] = settings_obj
        calls["ports"] = ports
        return 42

    monkeypatch.setattr(
        index_cmd_module,
        "get_cli_container",
        lambda: _FakeContainer(),
        raising=True,
    )
    monkeypatch.setattr(index_cmd_module, "run_cli_mutation", _fake_run_cli_mutation, raising=True)
    monkeypatch.setattr(
        "local_rag_backend.core.use_cases.index.rebuild_index_sync",
        _fake_rebuild_index_sync,
        raising=True,
    )

    result = CliRunner().invoke(index_cmd_module.rebuild_index_cmd)
    assert result.exit_code == 0, result.output
    assert "Rebuilt index with 42 vectors." in result.output
    assert calls["ports"] == "PORTS"
    assert calls["settings_obj"] is settings
    assert calls["bundle_called"] is True
    assert "build_embedder" in calls["bundle_kwargs"]
    assert calls["bundle_kwargs"]["use_wiring_defaults"] is True
    assert calls["mutation_kwargs"] == {}
