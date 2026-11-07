"""Probe an explicitly installed wheel outside the checkout, denying read-time writes.

Usage: uv run python scripts/smoke_ephemeral_snapshot_wheel.py \
  --rag-wheel /exact/rag.whl --python /fresh/wheel-env/bin/python \
  --work-dir /fresh/owned/acceptance
The caller installs the exact wheel; this script never selects/builds a package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast


def _fixture(root: Path, url: str) -> None:
    from sqlalchemy import create_engine

    from local_rag_backend.core.domain.types import DocId
    from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
        SentenceTransformerEmbedder,
    )
    from local_rag_backend.infrastructure.persistence.sql.base import Base
    from local_rag_backend.infrastructure.persistence.sql.models import Document, QaHistory
    from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
    from local_rag_backend.settings import Settings

    for mode in ("sparse", "dense", "dual", "hybrid"):
        case = root / mode
        case.mkdir()
        settings = Settings(
            sqlite_url=f"sqlite:///{case / 'live.sqlite3'}",
            data_dir=case / "live-data",
            retrieval_mode=mode,
            synthetic_embeddings=True,
            synthetic_embedding_dim=8,
            vector_backend="numpy",
            index_path=str(case / "live-vectors.bin"),
            id_map_path=str(case / "live-ids.json"),
            ephemeral_snapshot_dir=case / "frozen",
            ollama_enabled=True,
            ollama_base_url=url,
            openai_api_key=None,
            perf_metrics_out_path=str(case / "forbidden-perf.json"),
        )
        engine = create_engine(settings.sqlite_url)
        Base.metadata.create_all(engine)
        with engine.begin() as connection:
            connection.execute(
                Document.__table__.insert(),
                {
                    "doc_id": "one",
                    "external_id": "one",
                    "content": "alpha beta",
                    "scope": "public",
                    "snapshot_id": "synthetic-generation",
                    "metadata": {"kind": "sample"},
                },
            )
            connection.execute(
                QaHistory.__table__.insert(),
                {"question": "PRIVATE-HISTORY-SYNTHETIC", "answer": "excluded"},
            )
        engine.dispose()
        if mode in {"dense", "hybrid"}:
            embedder = SentenceTransformerEmbedder(settings_obj=settings)
            VectorStorage(
                settings.index_path,
                settings.id_map_path,
                dim=embedder.dim,
                backend="numpy",
                settings_obj=settings,
                embedding_identity=embedder.identity,
            ).rebuild([DocId("one")], embedder.embed(["alpha beta"]))
        (case / "config.json").write_text(json.dumps(settings.model_dump(mode="json")))
        if mode == "sparse":
            missing = settings.model_copy(update={"ephemeral_snapshot_dir": root / "missing"})
            (root / "missing-config.json").write_text(json.dumps(missing.model_dump(mode="json")))
        print(json.dumps({"fixture_prepared": mode}), flush=True)


def _probe(root: Path) -> None:
    from local_rag_backend import mcp_server
    from local_rag_backend.settings import load_settings_from_yaml

    if not Path(mcp_server.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError("Probe requires a wheel installed inside the supplied environment")

    settings = [
        load_settings_from_yaml(root / mode / "config.json")
        for mode in ("sparse", "dense", "dual", "hybrid")
    ]
    writes: list[str] = []
    denied = {
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.link",
        "os.symlink",
        "os.chmod",
        "os.chown",
        "os.truncate",
        "os.utime",
        "subprocess.Popen",
        "os.system",
    }
    active = True

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if not active:
            return
        forbidden = event in denied
        if event == "open":
            flags = arguments[2]
            forbidden = isinstance(flags, int) and bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
        if event == "sqlite3.connect":
            forbidden = arguments[0] != ":memory:"
        if forbidden:
            writes.append(event)
            raise AssertionError(f"Read operation attempted {event}")

    sys.addaudithook(audit)
    results = []
    try:
        for configured in settings:
            os.environ["RAG_CONFIG_PATH"] = str(root / configured.retrieval_mode / "config.json")
            mcp_server.get_settings.cache_clear()
            status = mcp_server.tool_status()
            health = cast("dict[str, object]", status["health"])
            assert health["snapshot_ready"] is True
            result = mcp_server.tool_ask_ephemeral(
                "alpha",
                filters=[
                    {"field": "scope", "values": ["public"]},
                    {"field": "snapshot_id", "values": ["synthetic-generation"]},
                ],
            )
            assert result["answer"] == "synthetic answer"
            sources = cast("list[dict[str, Any]]", result["sources"])
            assert sources[0]["document"]["id"] == "one"
            assert result["snapshot_id"] == health["snapshot_id"]
            results.append(
                {"mode": configured.retrieval_mode, "snapshot_id": result["snapshot_id"]}
            )
        os.environ["RAG_CONFIG_PATH"] = str(root / "missing-config.json")
        mcp_server.get_settings.cache_clear()
        assert (
            cast("dict[str, object]", mcp_server.tool_status()["health"])["snapshot_ready"] is False
        )
        try:
            mcp_server.tool_ask_ephemeral("alpha")
        except mcp_server.ToolOperationError:
            pass
        else:
            raise AssertionError("Missing snapshot was accepted")
        assert not writes, writes
    finally:
        active = False
    package_root = Path(mcp_server.__file__).resolve().parent
    package_files = {
        "local_rag_backend/" + str(path.relative_to(package_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in package_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    print(
        json.dumps(
            {
                "version": importlib.metadata.version("rag-prototype"),
                "module": str(Path(mcp_server.__file__).resolve()),
                "queries": results,
                "write_attempts": writes,
                "package_files": package_files,
            }
        )
    )


def _digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--rag-wheel", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("fixture", "probe"))
    parser.add_argument("--url")
    args = parser.parse_args()
    root = args.work_dir.absolute()
    if args.phase == "fixture":
        _fixture(root, args.url)
        return
    if args.phase == "probe":
        _probe(root)
        return
    if args.python is None:
        parser.error("--python must name a fresh environment with the exact wheel installed")
    if args.rag_wheel is None or not args.rag_wheel.is_file():
        parser.error("--rag-wheel must identify the exact installed artifact")
    root.mkdir(parents=False, exist_ok=False)
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(self.path)
            body = b'{"response":"synthetic answer"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OLLAMA_ENABLED": "true",
        "OLLAMA_BASE_URL": f"http://127.0.0.1:{server.server_port}",
    }
    env.pop("PYTHONPATH", None)
    script = str(Path(__file__).resolve())
    command = [str(args.python.absolute()), script, "--work-dir", str(root)]
    try:
        subprocess.run(
            [*command, "--phase", "fixture", "--url", f"http://127.0.0.1:{server.server_port}"],
            check=True,
            cwd=root,
            env=env,
            timeout=300,
        )
        for mode in ("sparse", "dense", "dual", "hybrid"):
            case = root / mode
            subprocess.run(
                [
                    str(args.python.absolute()),
                    "-m",
                    "local_rag_backend.cli",
                    "snapshot",
                    str(case / "frozen"),
                ],
                check=True,
                cwd=root,
                env={**env, "RAG_CONFIG_PATH": str(case / "config.json")},
                stdout=subprocess.PIPE,
                timeout=180,
            )
            assert (
                b"PRIVATE-HISTORY-SYNTHETIC"
                not in (case / "frozen" / "documents.sqlite3").read_bytes()
            )
        before = _digests(root)
        completed = subprocess.run(
            [*command, "--phase", "probe"],
            check=True,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        receipt = json.loads(completed.stdout)
        with zipfile.ZipFile(args.rag_wheel) as wheel:
            expected = {
                name: hashlib.sha256(wheel.read(name)).hexdigest()
                for name in wheel.namelist()
                if name.startswith("local_rag_backend/") and not name.endswith("/")
            }
        assert receipt.pop("package_files") == expected, (
            "Installed package bytes differ from the explicit wheel"
        )
        receipt["package_files_verified"] = len(expected)
        receipt["wheel_sha256"] = hashlib.sha256(args.rag_wheel.read_bytes()).hexdigest()
        assert _digests(root) == before
        assert requests == ["/api/generate"] * 4
        receipt["provider_calls"] = len(requests)
        receipt["files_unchanged"] = True
        (root / "acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
