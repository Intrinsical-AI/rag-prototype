"""Validate exact RepoGPT and chat-adapter wheels against an installed RAG wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

CONSUMER = r"""
import importlib
import json
import subprocess
import sys
from dataclasses import replace
from importlib import metadata, resources
from pathlib import Path

from local_rag_backend.composition.container import AppContainer
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.core.services.canonical_import_transport import (
    resolve_canonical_import_replace_scope, validate_canonical_import_payload,
)
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever
from local_rag_backend.settings import get_settings
from rag_adapters.codec import load_archive, save_archive
from rag_adapters.projectors import PROJECTORS, apply_projections

root = Path.cwd()
venv = Path(sys.prefix).resolve()
versions = json.loads(sys.argv[1])
for distribution, module in (
    ("repogpt", "repogpt"), ("rag-adapters", "rag_adapters"),
    ("rag-prototype", "local_rag_backend"),
):
    origin = Path(importlib.import_module(module).__file__).resolve()
    assert origin.is_relative_to(venv), origin
    assert metadata.version(distribution) == versions[distribution]
for module, schema in (
    ("repogpt", "code-units-v5.schema.json"),
    ("rag_adapters", "chat-archive-v2.schema.json"),
    ("rag_adapters", "rag-import-chat-v2.schema.json"),
):
    assert resources.files(module).joinpath("schemas", schema).is_file()

def run(command, expected=0):
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    assert completed.returncode == expected, (command, completed.returncode, completed.stderr)
    return completed

fixture = root / "repo"
fixture.mkdir()
(fixture / "sample.py").write_text(
    "# installed artifact fixture\ndef wheel_helper():\n    return 'boundary_needle'\n",
    encoding="utf-8",
)
payload_path = root / "repogpt.json"
run([sys.executable, "-m", "repogpt.app.cli", "--emit", "code-units", "--repo-key",
     "wheel-fixture", "--replace-scope", "--include-tests", "-o", str(payload_path), str(fixture)])
payload = json.loads(payload_path.read_text())
assert payload["schema_version"] == "5" and payload["replace_scope"] is True
validate_canonical_import_payload(payload)
rag_cli = str(venv / "bin" / "rag-import-canonical")
run([rag_cli, "--json", str(payload_path)])
run([rag_cli, "--json", str(payload_path)])

chat = root / "chat"
chat.mkdir()
(chat / "_chat.txt").write_text(
    "[1/8/26, 10:00:00] Synthetic: august_needle\n"
    "[1/9/26, 10:00:00] Synthetic: september_needle\n"
    "[1/10/26, 10:00:00] Synthetic: excluded\n", encoding="utf-8",
)
archive_path = root / "archive" / "chat_archive.json"
run([sys.executable, "-m", "rag_adapters.cli", "ingest", str(chat), "-o",
     str(archive_path.parent), "--tz", "UTC", "--date-order", "dmy", "--chat-key", "wheel-chat"])
archive = load_archive(archive_path)
archive.messages[-1].role = "system"
failure = {"code": "projection_failed", "message_id": archive.messages[-1].message_id}
archive.failures.append(failure)
save_archive(archive, archive_path)
export = root / "exports"
run([sys.executable, "-m", "rag_adapters.cli", "export-rag", str(archive_path), "-o", str(export)],
    expected=2)
chat_paths = sorted(export.glob("*.json"))
assert len(chat_paths) == 2
for path in chat_paths:
    chat_payload = json.loads(path.read_text())
    assert chat_payload["failures"] == [failure] and chat_payload["stats"]["failed_files"] == 1
    assert not any(key in chat_payload for key in ("kind", "schema_version", "repo_key"))
    assert chat_payload["replace_scope"] is False
    validated = validate_canonical_import_payload(chat_payload)
    assert resolve_canonical_import_replace_scope(validated) is False
    run([rag_cli, "--json", str(path)])
    run([rag_cli, "--json", str(path)])
    run([rag_cli, "--json", str(path), "--replace-scope"], expected=1)

media_chat = root / "media-chat"
media_chat.mkdir()
(media_chat / "first.pdf").write_bytes(b"synthetic selected bytes")
(media_chat / "unused.txt").write_bytes(b"synthetic unselected bytes")
(media_chat / "_chat.txt").write_text(
    "[1/9/26, 10:00:00] Synthetic: first.pdf (archivo adjunto)\n"
    "[1/9/26, 10:01:00] Synthetic: unused.txt (archivo adjunto)\n", encoding="utf-8",
)
bundle = root / "media-bundle"
run([sys.executable, "-m", "rag_adapters.cli", "ingest", str(media_chat), "-o", str(bundle),
     "--tz", "UTC", "--date-order", "dmy", "--chat-key", "wheel-media", "--copy-media"])
media_archive = load_archive(bundle / "chat_archive.json")
selected = next(ref for ref in media_archive.media if ref.filename == "first.pdf")
unused = next(ref for ref in media_archive.media if ref.filename == "unused.txt")
PROJECTORS["pdf"] = replace(PROJECTORS["pdf"], package="rag-adapters", extract=lambda _p: "stub")
(bundle / unused.bundle_rel_path).write_bytes(b"changed unselected bytes")
apply_projections(media_archive, bundle, requested=["pdf"], cache_dir=root / "cache",
                  use_bundle=True, verify_all=False)
assert len(media_archive.projections) == 1
(bundle / selected.bundle_rel_path).write_bytes(b"changed selected bytes")
try:
    apply_projections(media_archive, bundle, requested=["pdf"], cache_dir=root / "cache",
                      use_bundle=True, verify_all=False)
except ValueError as error:
    assert "hash mismatch" in str(error)
else:
    raise AssertionError("selected bytes were not checked before projection reuse")

container = AppContainer.from_settings(get_settings())
container.initialize()
try:
    repo = container.doc_repo_factory()
    documents = list(repo.get_all_documents())
    expected_docs = [doc for doc in payload["documents"] if doc["content"].strip()]
    assert len(documents) == len(expected_docs) + 2
    retriever = LocalSplitSearchRetriever(doc_repo=repo)
    found = retriever.retrieve(RetrievalRequest(
        query="boundary_needle", top_k=1, mode="sparse",
        filters=(RetrievalFilter("metadata.unit_type", ("function",)),),
    ))
    assert found.items and found.items[0].document.metadata["repo_key"] == "wheel-fixture"
    assert retriever.retrieve(RetrievalRequest(query="september_needle", top_k=1, mode="sparse")).items
    overlong = json.loads(json.dumps(payload))
    overlong["documents"][0]["external_id"] = "x" * 513
    try:
        validate_canonical_import_payload(overlong)
    except ValueError:
        pass
    else:
        raise AssertionError("current canonical external-ID limit was not enforced")
    assert len(repo.get_all_documents()) == len(documents)
finally:
    container.close()
print(json.dumps({"versions": versions, "repo_documents": len(expected_docs),
                  "chat_months": len(chat_paths), "stored_documents": len(documents),
                  "replay_idempotent": True, "partial_replacement_refused": True,
                  "origins_in_fresh_venv": True, "consumer_limits_retained": True,
                  "selective_bundle_verified": True, "real_extractors": False}, sort_keys=True))
"""


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        command, cwd=cwd, env=env, check=False, capture_output=True, text=True, timeout=120
    )
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def _wheel(path: Path, name: str) -> dict[str, str]:
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError(f"Expected an explicit wheel file for {name}: {path}")
    with ZipFile(path) as wheel:
        manifests = [item for item in wheel.namelist() if item.endswith(".dist-info/METADATA")]
        if len(manifests) != 1:
            raise ValueError(f"Expected one distribution metadata file: {path}")
        metadata = BytesParser().parsebytes(wheel.read(manifests[0]))
    distribution = str(metadata["Name"]).lower().replace("_", "-")
    if distribution != name:
        raise ValueError(f"Expected {name}, found {distribution}: {path}")
    return {
        "path": str(path),
        "version": str(metadata["Version"]),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repogpt-wheel", type=Path, required=True)
    parser.add_argument("--adapters-wheel", type=Path, required=True)
    parser.add_argument("--rag-wheel", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required")
    wheels = {
        name: _wheel(path.resolve(), name)
        for name, path in (
            ("repogpt", args.repogpt_wheel),
            ("rag-adapters", args.adapters_wheel),
            ("rag-prototype", args.rag_wheel),
        )
    }
    env = os.environ.copy()
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "RAG_CONFIG_PATH",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="canonical-wheels-", dir=args.scratch_root) as temp:
        root = Path(temp)
        venv = root / "venv"
        _run([uv, "venv", "--python", sys.executable, str(venv)], cwd=root, env=env)
        python = venv / "bin/python"
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                *(item["path"] for item in wheels.values()),
            ],
            cwd=root,
            env=env,
        )
        _run([uv, "pip", "check", "--python", str(python)], cwd=root, env=env)
        config = root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "sqlite_url": f"sqlite:///{root / 'db.sqlite3'}",
                    "data_dir": str(root / "data"),
                    "index_path": str(root / "index.npy"),
                    "id_map_path": str(root / "ids.json"),
                    "retrieval_mode": "sparse",
                    "openai_api_key": None,
                    "ollama_enabled": False,
                    "synthetic_embeddings": True,
                    "synthetic_embedding_dim": 4,
                    "disable_embedding_cache": True,
                }
            ),
            encoding="utf-8",
        )
        env["RAG_CONFIG_PATH"] = str(config)
        consumer = root / "consumer.py"
        consumer.write_text(CONSUMER, encoding="utf-8")
        result = _run(
            [str(python), str(consumer), json.dumps({k: v["version"] for k, v in wheels.items()})],
            cwd=root,
            env=env,
        )
        print(json.dumps({"wheels": wheels, "boundary": json.loads(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
