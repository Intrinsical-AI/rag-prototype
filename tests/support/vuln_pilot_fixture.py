from __future__ import annotations

import importlib.util
from typing import Any

from support.external_paths import configured_file

VULN_PILOT_PREPARED = configured_file("VULN_PILOT_PREPARED")
VULNS_INGEST_SCRIPT = configured_file("VULNS_INGEST_SCRIPT")

__all__ = ["VULNS_INGEST_SCRIPT", "VULN_PILOT_PREPARED", "load_vulns_ingest_module"]


def load_vulns_ingest_module() -> Any:
    if VULNS_INGEST_SCRIPT is None:
        raise RuntimeError("VULNS_INGEST_SCRIPT is required for this external E2E fixture")
    spec = importlib.util.spec_from_file_location("vulns_ingest_rag", VULNS_INGEST_SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load script module: {VULNS_INGEST_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
