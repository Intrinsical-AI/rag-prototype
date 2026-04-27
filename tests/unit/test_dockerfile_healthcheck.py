from __future__ import annotations

from pathlib import Path


def test_dockerfile_healthcheck_targets_live_health_route() -> None:
    dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    assert "http://localhost:8000/healthz" in text
    assert "http://localhost:8000/api/health" not in text


def test_dockerfile_local_copy_sources_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = repo_root / "Dockerfile"

    missing_sources: list[str] = []
    for raw_line in dockerfile.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        parts = line.split()
        for source in parts[1:-1]:
            if source.startswith("--"):
                continue
            if not (repo_root / source).exists():
                missing_sources.append(source)

    assert missing_sources == []
