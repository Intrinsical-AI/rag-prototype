from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from support.external_paths import configured_directory

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REPOGPT_ROOT = Path(os.environ.get("REPOGPT_ROOT", WORKSPACE_ROOT / "RepoGPT"))
REPOGPT_FIXTURE_REPO = configured_directory("REPOGPT_FIXTURE_REPO")


def _preparation_message(root: Path) -> str:
    return (
        f"Prepare RepoGPT in {root} with `uv sync --frozen --extra dev`; "
        "REPOGPT_ROOT must select its checkout with a working .venv/bin/python."
    )


def resolve_repogpt_python(root: Path, *, configured: bool) -> Path | None:
    if not root.exists() and not configured:
        return None
    python_executable = root / ".venv" / "bin" / "python"
    if not root.is_dir() or not os.access(python_executable, os.X_OK):
        raise RuntimeError(_preparation_message(root))
    return python_executable


# Skip only an absent, unconfigured optional integration. Broken installations must fail.
REPOGPT_CLI_AVAILABLE = (
    resolve_repogpt_python(REPOGPT_ROOT, configured="REPOGPT_ROOT" in os.environ) is not None
)


def emit_repogpt_code_units(
    *,
    payload_path: Path,
    repo_path: Path | None = REPOGPT_FIXTURE_REPO,
) -> dict[str, object]:
    if repo_path is None:
        raise RuntimeError("REPOGPT_FIXTURE_REPO is required for this external E2E fixture")
    preparation = _preparation_message(REPOGPT_ROOT)
    python_executable = resolve_repogpt_python(REPOGPT_ROOT, configured=True)
    env = dict(os.environ)
    repogpt_src = REPOGPT_ROOT / "src"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repogpt_src}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repogpt_src)
    )
    completed = subprocess.run(
        [
            str(python_executable),
            "-m",
            "repogpt.app.cli",
            "--emit",
            "code-units",
            "-o",
            str(payload_path),
            str(repo_path),
        ],
        cwd=REPOGPT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"RepoGPT code-units emission failed. {preparation}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(payload_path.read_text(encoding="utf-8"))
