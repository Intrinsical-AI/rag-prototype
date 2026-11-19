from __future__ import annotations

import tomllib
from pathlib import Path

from click.testing import CliRunner

from local_rag_backend import cli as cli_module


def test_canonical_command_tree_has_one_owner() -> None:
    expected = {
        "docs": {"bootstrap", "ingest", "import", "mutate"},
        "index": {"rebuild"},
        "query": {"search", "ask"},
        "serve": {"http", "mcp"},
        "eval": {"run", "batch", "compare"},
    }
    assert set(cli_module.cli.commands) == set(expected) | {"status", "snapshot"}
    for group, commands in expected.items():
        assert set(cli_module.cli.commands[group].commands) == commands


def test_installed_console_surface_is_only_rag() -> None:
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert project["project"]["scripts"] == {"rag": "local_rag_backend.cli:cli"}


def test_removed_top_level_commands_are_rejected() -> None:
    for name in ("server", "bootstrap", "mutate-docs", "import-canonical", "eval-batch", "ingest"):
        result = CliRunner().invoke(cli_module.cli, [name])
        assert result.exit_code == 2
        assert "No such command" in result.output
