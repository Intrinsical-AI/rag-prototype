from __future__ import annotations

from pathlib import Path


def test_dockerfile_healthcheck_targets_live_health_route() -> None:
    dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    assert "http://localhost:8000/healthz" in text
    assert "http://localhost:8000/api/health" not in text
