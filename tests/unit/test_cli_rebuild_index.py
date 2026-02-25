from __future__ import annotations

from click.testing import CliRunner

from local_rag_backend.cli_commands import index as index_cmd_module
from local_rag_backend.settings import settings


def test_rebuild_index_cli_delegates_to_app_service(monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    calls: dict[str, object] = {}

    def _fake_build_dense_embedder():
        calls["build_dense_embedder_called"] = True

        class _DummyEmbedder:
            dim = 4

        return _DummyEmbedder()

    def _fake_run_cli_mutation(operation, **kwargs):
        calls["mutation_kwargs"] = kwargs
        return operation()

    def _fake_build_index_mutation_ports(*, build_embedder, **kwargs):
        embedder = build_embedder()
        calls["embedder_dim"] = embedder.dim
        calls["ports_kwargs"] = kwargs
        return "PORTS"

    def _fake_rebuild_index_sync(*, settings_obj, ports):
        calls["settings_obj"] = settings_obj
        calls["ports"] = ports
        return 42

    monkeypatch.setattr(
        index_cmd_module, "build_dense_embedder", _fake_build_dense_embedder, raising=True
    )
    monkeypatch.setattr(index_cmd_module, "run_cli_mutation", _fake_run_cli_mutation, raising=True)
    monkeypatch.setattr(
        "local_rag_backend.composition.wiring.mutation_ports.build_index_mutation_ports",
        _fake_build_index_mutation_ports,
        raising=True,
    )
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
    assert calls["embedder_dim"] == 4
    assert calls["build_dense_embedder_called"] is True
    assert calls["mutation_kwargs"] == {}
