from click.testing import CliRunner

from local_rag_backend.cli_commands import server as server_module
from local_rag_backend.settings import get_settings

settings = get_settings()


def test_server_command_passes_runtime_settings_to_uvicorn(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "fixture-key")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(settings, "app_host", "127.0.0.2", raising=False)
    monkeypatch.setattr(settings, "app_port", 8123, raising=False)
    monkeypatch.setattr(settings, "debug", True, raising=False)
    monkeypatch.setattr(settings, "log_level", "WARNING", raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(
        "uvicorn.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = CliRunner().invoke(server_module.server_cmd)

    assert result.exit_code == 0
    assert calls == [
        (
            ("local_rag_backend.http.main:app",),
            {
                "host": "127.0.0.2",
                "port": 8123,
                "reload": True,
                "workers": 1,
                "log_level": "warning",
            },
        )
    ]
    assert "Development (reload)" in result.output
    assert "Retrieval: sparse" in result.output


def test_production_workers_and_reload_override(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "debug", True)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(server_module.server_cmd, ["--workers", "4", "--no-reload"])
    assert result.exit_code == 0
    assert calls[0]["workers"] == 4
    assert calls[0]["reload"] is False
    assert (
        CliRunner().invoke(server_module.server_cmd, ["--workers", "4", "--reload"]).exit_code == 2
    )
