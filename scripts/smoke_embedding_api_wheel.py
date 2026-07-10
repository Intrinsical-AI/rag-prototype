"""Build and smoke-test the public embedding API from an isolated wheel install."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_CONFIG = """\
openai_api_key: null
st_embedding_model: wheel-synthetic
synthetic_embeddings: true
synthetic_embedding_dim: 4
synthetic_embedding_fail_rate: 0.0
synthetic_embedding_jitter_min_ms: 0.0
synthetic_embedding_jitter_max_ms: 0.0
data_dir: data
embedding_cache_db_path: data/cache.sqlite3
disable_embedding_cache: false
"""

CONSUMER_SCRIPT = """\
from __future__ import annotations

import importlib.util
import json
import sys
from importlib import resources

from local_rag_backend.integrations.embeddings import (
    EmbeddingService,
    create_embedding_service,
)

service = create_embedding_service(sys.argv[1] if len(sys.argv) > 1 else None)
status = service.status()
vectors = service.embed(["alpha", "beta", "alpha"])
payload = {
    "is_service": isinstance(service, EmbeddingService),
    "module_path": str(resources.files("local_rag_backend")),
    "py_typed": resources.files("local_rag_backend").joinpath("py.typed").is_file(),
    "sentence_transformers_installed": (
        importlib.util.find_spec("sentence_transformers") is not None
    ),
    "status": status,
    "vectors": vectors,
}
print(json.dumps(payload, allow_nan=False, sort_keys=True))
"""


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        output = "\n".join(value.strip() for value in (error.stdout, error.stderr) if value.strip())
        detail = f"\n{output}" if output else ""
        raise RuntimeError(
            f"Command failed ({error.returncode}): {shlex.join(command)}{detail}"
        ) from error


def _clean_process_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "RAG_CONFIG_PATH",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
    ):
        env.pop(key, None)
    return env


def _assert_payload(payload: dict[str, Any], *, venv_dir: Path) -> None:
    status = payload["status"]
    vectors = payload["vectors"]
    assert payload["is_service"] is True
    assert payload["py_typed"] is True
    assert payload["sentence_transformers_installed"] is False
    assert str(venv_dir) in str(payload["module_path"])
    assert status["provider"] == "sentence_transformers"
    assert status["model"] == "wheel-synthetic"
    assert status["dimension"] == 4
    assert status["synthetic"] is True
    assert status["cache_enabled"] is True
    assert len(vectors) == 3
    assert all(len(vector) == 4 for vector in vectors)
    assert vectors[0] == vectors[2]


def main() -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the wheel embedding API smoke test")

    with tempfile.TemporaryDirectory(prefix="rag-embedding-wheel-smoke-") as temp_dir:
        root = Path(temp_dir)
        dist_dir = root / "dist"
        venv_dir = root / "venv"
        runtime_dir = root / "runtime"
        outside_dir = root / "outside"
        runtime_dir.mkdir()
        outside_dir.mkdir()
        config_path = runtime_dir / "config.yaml"
        config_path.write_text(SYNTHETIC_CONFIG, encoding="utf-8")
        consumer_path = root / "consumer.py"
        consumer_path.write_text(CONSUMER_SCRIPT, encoding="utf-8")

        clean_env = _clean_process_env()
        _run([uv, "build", "--out-dir", str(dist_dir)], cwd=REPO_ROOT, env=clean_env)
        wheels = sorted(dist_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one wheel, found: {wheels}")

        _run([uv, "venv", str(venv_dir)], cwd=root, env=clean_env)
        python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run(
            [uv, "pip", "install", "--python", str(python), str(wheels[0])],
            cwd=root,
            env=clean_env,
        )

        explicit = _run(
            [str(python), str(consumer_path), str(config_path)],
            cwd=outside_dir,
            env=clean_env,
        )
        explicit_payload = json.loads(explicit.stdout)
        _assert_payload(explicit_payload, venv_dir=venv_dir)

        env_selected = clean_env.copy()
        env_selected["RAG_CONFIG_PATH"] = str(config_path)
        environment = _run(
            [str(python), str(consumer_path)],
            cwd=outside_dir,
            env=env_selected,
        )
        environment_payload = json.loads(environment.stdout)
        _assert_payload(environment_payload, venv_dir=venv_dir)

        if explicit_payload["status"] != environment_payload["status"]:
            raise RuntimeError("Explicit and environment-selected status differ")
        if explicit_payload["vectors"] != environment_payload["vectors"]:
            raise RuntimeError("Explicit and environment-selected vectors differ")
        cache_path = Path(str(explicit_payload["status"]["cache_db_path"]))
        if not cache_path.is_file():
            raise RuntimeError(f"Expected embedding cache was not created: {cache_path}")

        print(
            json.dumps(
                {
                    "dimension": explicit_payload["status"]["dimension"],
                    "model_key": explicit_payload["status"]["model_key"],
                    "provider": explicit_payload["status"]["provider"],
                    "py_typed": explicit_payload["py_typed"],
                    "sentence_transformers_installed": explicit_payload[
                        "sentence_transformers_installed"
                    ],
                    "wheel": wheels[0].name,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
