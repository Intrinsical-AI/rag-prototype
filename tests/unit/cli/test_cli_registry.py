from __future__ import annotations

import pytest

from local_rag_backend import cli as cli_module


def test_cli_registers_all_expected_commands() -> None:
    expected = {
        "server",
        "rebuild-index",
        "mutate-docs",
        "import-canonical",
        "bootstrap",
        "status",
        "eval",
        "eval-batch",
        "ingest",
    }
    assert expected <= set(cli_module.cli.commands.keys())
    assert "upsert-docs" not in cli_module.cli.commands
    assert "delete-docs" not in cli_module.cli.commands
    assert "delete-external-ids" not in cli_module.cli.commands


def test_cli_does_not_register_legacy_mutation_commands() -> None:
    legacy = {
        "upsert-docs",
        "delete-docs",
        "delete-external-ids",
    }
    assert legacy.isdisjoint(cli_module.cli.commands.keys())


@pytest.mark.parametrize(
    ("wrapper_name", "expected_args"),
    [
        ("rag_server", ["server"]),
        ("rag_bootstrap", ["bootstrap", "--flag"]),
        ("rag_status", ["status", "--flag"]),
        ("rag_eval", ["eval", "--flag"]),
        ("rag_eval_batch", ["eval-batch", "--flag"]),
        ("rag_rebuild_index", ["rebuild-index", "--flag"]),
        ("rag_mutate_docs", ["mutate-docs", "--flag"]),
        ("rag_import_canonical", ["import-canonical", "--flag"]),
        ("rag_ingest", ["ingest", "--flag"]),
    ],
)
def test_entrypoint_wrappers_delegate_to_click_main(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_name: str,
    expected_args: list[str],
) -> None:
    calls: list[tuple[list[str], bool]] = []

    def _fake_main(*, args: list[str], standalone_mode: bool) -> None:
        calls.append((list(args), standalone_mode))

    monkeypatch.setattr(cli_module.cli, "main", _fake_main, raising=True)
    monkeypatch.setattr(cli_module.sys, "argv", ["prog", "--flag"], raising=True)

    wrapper = getattr(cli_module, wrapper_name)
    wrapper()

    assert calls == [(expected_args, False)]
