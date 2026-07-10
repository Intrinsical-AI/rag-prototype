"""Optional RAGAS adapter boundary.

The adapter keeps RAGAS-specific objects outside the core evaluation model.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from local_rag_backend.core.services.evaluation import eval_result_report_to_json
from local_rag_backend.core.services.evaluation_models import EvalResult


class EvalAdapterUnavailableError(RuntimeError):
    """Raised when an optional evaluation adapter dependency is unavailable."""


class RagasEvaluationAdapter:
    def __init__(self) -> None:
        self._ragas: Any | None = None

    def _load_ragas(self) -> Any:
        if self._ragas is None:
            try:
                self._ragas = import_module("ragas")
            except ImportError as exc:
                raise EvalAdapterUnavailableError(
                    "RAGAS support requires installing ragas in the runtime environment."
                ) from exc
        return self._ragas

    def ensure_available(self) -> None:
        self._load_ragas()

    def build_report_payload(self, result: EvalResult) -> dict[str, Any]:
        payload = eval_result_report_to_json(result)
        payload["adapter"] = "ragas"
        return payload
