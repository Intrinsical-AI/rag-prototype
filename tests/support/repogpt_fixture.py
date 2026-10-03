from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from support.external_paths import configured_directory

REPOGPT_ROOT = configured_directory("REPOGPT_ROOT")
CANONICAL_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/repogpt_code_units_v5.json"
REPOGPT_SOURCE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/repogpt_eval_repo"


def _preparation_message(root: Path) -> str:
    return (
        f"Prepare RepoGPT in {root} with `uv sync --frozen --extra dev`; "
        "REPOGPT_ROOT must select its checkout with a working .venv/bin/python."
    )


def resolve_repogpt_python(root: Path) -> Path:
    python_executable = root / ".venv" / "bin" / "python"
    if not root.is_dir() or not os.access(python_executable, os.X_OK):
        raise RuntimeError(_preparation_message(root))
    return python_executable


def load_repogpt_payload(
    *,
    payload_path: Path,
    repo_path: Path = REPOGPT_SOURCE_FIXTURE,
) -> dict[str, object]:
    if REPOGPT_ROOT is None:
        raw = CANONICAL_FIXTURE.read_text(encoding="utf-8")
        payload_path.write_text(raw, encoding="utf-8")
        return json.loads(raw)
    preparation = _preparation_message(REPOGPT_ROOT)
    python_executable = resolve_repogpt_python(REPOGPT_ROOT)
    env = dict(os.environ)
    repogpt_src = REPOGPT_ROOT / "src"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repogpt_src}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repogpt_src)
    )
    try:
        completed = subprocess.run(
            [
                str(python_executable),
                "-m",
                "repogpt.app.cli",
                "--emit",
                "code-units",
                "--repo-key",
                "repogpt_eval_repo",
                "--replace-scope",
                "--include-tests",
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
    except OSError as exc:
        raise RuntimeError(f"RepoGPT code-units launch failed. {preparation}") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"RepoGPT code-units emission failed. {preparation}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(payload_path.read_text(encoding="utf-8"))
