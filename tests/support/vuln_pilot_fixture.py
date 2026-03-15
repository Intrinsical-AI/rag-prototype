from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SYNERGY_ROOT = WORKSPACE_ROOT / "synergy"
VULN_PILOT_PREPARED = SYNERGY_ROOT / "vuln_pilot" / "prepared" / "pilot_small_v1.jsonl"


def load_vulns_ingest_module() -> Any:
    script_path = SYNERGY_ROOT / "scripts" / "vulns_ingest_rag.py"
    spec = importlib.util.spec_from_file_location("vulns_ingest_rag", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
