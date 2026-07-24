from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from support.external_paths import configured_directory

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REPOGPT_ROOT = Path(os.environ.get("REPOGPT_ROOT", WORKSPACE_ROOT / "RepoGPT"))
REPOGPT_CLI_AVAILABLE = REPOGPT_ROOT.is_dir()
REPOGPT_FIXTURE_REPO = configured_directory("REPOGPT_FIXTURE_REPO")


def emit_repogpt_code_units(
    *,
    payload_path: Path,
    repo_path: Path | None = REPOGPT_FIXTURE_REPO,
) -> dict[str, object]:
    if repo_path is None:
        raise RuntimeError("REPOGPT_FIXTURE_REPO is required for this external E2E fixture")
    env = dict(os.environ)
    repogpt_src = REPOGPT_ROOT / "src"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repogpt_src}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repogpt_src)
    )
    python_executable = REPOGPT_ROOT / ".venv" / "bin" / "python"
    if not python_executable.exists():
        python_executable = Path(sys.executable)
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
            "RepoGPT code-units emission failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(payload_path.read_text(encoding="utf-8"))
