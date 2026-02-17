from __future__ import annotations

from typing import Any


def _hooks() -> Any:
    from local_rag_backend import cli as cli_module

    return cli_module


def _reset_if_mutated(*, mutation_attempted: bool) -> None:
    if mutation_attempted:
        _hooks()._reset_rag_service_best_effort()

