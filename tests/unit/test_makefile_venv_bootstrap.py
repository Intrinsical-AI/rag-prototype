from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"


def _create_fake_uv(tmp_path: Path) -> tuple[Path, Path]:
    fake_uv = tmp_path / "fake-uv"
    log_path = tmp_path / "fake-uv.json"
    fake_uv.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import os\n"
        "import shutil\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "Path(os.environ['FAKE_UV_LOG']).write_text(json.dumps(args), encoding='utf-8')\n"
        "if not args or args[0] != 'venv':\n"
        "    raise SystemExit(2)\n"
        "target = Path(args[-1])\n"
        "if '--clear' in args and target.exists():\n"
        "    shutil.rmtree(target)\n"
        "(target / 'bin').mkdir(parents=True, exist_ok=True)\n"
        "(target / 'pyvenv.cfg').write_text('generated = true\\n', encoding='utf-8')\n"
        "python = target / 'bin' / 'python'\n"
        "python.write_text('#!/bin/sh\\nexit 0\\n', encoding='utf-8')\n"
        "python.chmod(0o755)\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    return fake_uv, log_path


def _run_make_venv(tmp_path: Path, venv_dir: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_uv, log_path = _create_fake_uv(tmp_path)
    env = os.environ.copy()
    env["FAKE_UV_LOG"] = str(log_path)
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-f",
            str(MAKEFILE),
            f"VENV_DIR={venv_dir}",
            f"UV={fake_uv}",
            "venv",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, log_path


def test_make_venv_creates_missing_environment(tmp_path):
    venv_dir = tmp_path / "missing-venv"

    result, log_path = _run_make_venv(tmp_path, venv_dir)

    assert result.returncode == 0, result.stderr
    assert json.loads(log_path.read_text(encoding="utf-8")) == ["venv", str(venv_dir)]
    assert (venv_dir / ".python-stamp").is_file()


def test_make_venv_accepts_usable_unstamped_environment(tmp_path):
    venv_dir = tmp_path / "usable-venv"
    python = venv_dir / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)

    result, log_path = _run_make_venv(tmp_path, venv_dir)

    assert result.returncode == 0, result.stderr
    assert not log_path.exists()
    assert (venv_dir / ".python-stamp").is_file()


def test_make_venv_recreates_only_broken_generated_environment(tmp_path):
    venv_dir = tmp_path / "broken-venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "pyvenv.cfg").write_text("generated = true\n", encoding="utf-8")
    (venv_dir / "bin" / "python").symlink_to(tmp_path / "missing-python")

    result, log_path = _run_make_venv(tmp_path, venv_dir)

    assert result.returncode == 0, result.stderr
    assert json.loads(log_path.read_text(encoding="utf-8")) == [
        "venv",
        "--clear",
        str(venv_dir),
    ]
    assert (venv_dir / "bin" / "python").is_file()
    assert (venv_dir / ".python-stamp").is_file()


def test_make_venv_refuses_unmarked_existing_directory(tmp_path):
    venv_dir = tmp_path / "not-a-venv"
    venv_dir.mkdir()
    sentinel = venv_dir / "keep-me"
    sentinel.write_text("user data\n", encoding="utf-8")

    result, log_path = _run_make_venv(tmp_path, venv_dir)

    assert result.returncode != 0
    assert "Refusing to replace existing non-virtualenv directory" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "user data\n"
    assert not log_path.exists()
