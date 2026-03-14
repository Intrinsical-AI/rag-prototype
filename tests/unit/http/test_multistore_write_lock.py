from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from dataclasses import dataclass

import pytest

from local_rag_backend.composition import factory
from local_rag_backend.http.routers import docs as docs_router
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_concurrent_mutations_are_serialized_and_keep_sql_vector_consistent(
    asgi_client, in_memory_sqlite, monkeypatch
):
    @dataclass(frozen=True)
    class _UpsertDoc:
        external_id: str
        content: str
        source_id: str | None = None
        scope: str | None = None
        snapshot_id: str | None = None
        metadata: dict[str, object] | None = None
        chunk_dedup_sha256: str | None = None

    @dataclass(frozen=True)
    class _UpsertResult:
        external_id: str
        id: int
        action: str
        content_changed: bool

    class FakeRepo:
        UpsertDoc = _UpsertDoc

        _lock = threading.Lock()
        _next_id = 1
        _by_external_id: dict[str, tuple[int, str, str]] = {}
        _a_sql_done = threading.Event()

        def snapshot_by_external_ids(self, external_ids):
            snapshots = []
            with self._lock:
                for ext in external_ids:
                    row = self._by_external_id.get(ext)
                    if row is None:
                        continue
                    doc_id, content, content_sha256 = row
                    snapshots.append(
                        {
                            "id": str(doc_id),
                            "external_id": ext,
                            "content": content,
                            "content_sha256": content_sha256,
                        }
                    )
            return snapshots

        def hard_delete_by_external_ids(self, external_ids):
            with self._lock:
                for ext in external_ids:
                    self._by_external_id.pop(ext, None)

        def restore_from_snapshots(self, snapshots):
            with self._lock:
                for snap in snapshots:
                    ext = str(snap["external_id"])
                    doc_id = int(snap["id"])
                    content = str(snap["content"])
                    content_sha256 = str(snap["content_sha256"])
                    self._by_external_id[ext] = (doc_id, content, content_sha256)
                    self._next_id = max(self._next_id, doc_id + 1)

        def get_tombstoned_external_ids(self, external_ids):
            return set()

        def upsert_documents_by_external_id(self, items):
            results = []
            changed = []
            updated_ids = []

            with self._lock:
                for item in items:
                    ext = item.external_id
                    content = item.content.strip()
                    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    row = self._by_external_id.get(ext)
                    if row is None:
                        doc_id = self._next_id
                        self._next_id += 1
                        self._by_external_id[ext] = (doc_id, content, sha)
                        results.append(_UpsertResult(ext, doc_id, "inserted", True))
                        changed.append((doc_id, content))
                        if content == "A":
                            self._a_sql_done.set()
                        continue

                    doc_id, old_content, old_sha = row
                    if content == "B":
                        self._a_sql_done.wait(timeout=2)
                    content_changed = old_sha != sha or old_content != content
                    if content_changed:
                        self._by_external_id[ext] = (doc_id, content, sha)
                        results.append(_UpsertResult(ext, doc_id, "updated", True))
                        changed.append((doc_id, content))
                        updated_ids.append(doc_id)
                    else:
                        results.append(_UpsertResult(ext, doc_id, "unchanged", False))

            return results, changed, updated_ids

    class FakeEmbedder:
        dim = 1

        def embed(self, texts):
            return [[1.0 if "A" in text else 2.0] for text in texts]

    class FakeVec:
        def __init__(self):
            self.by_id: dict[int, list[float]] = {}
            self._lock = threading.Lock()

        def apply_delta_atomic(self, *, delete_ids, upserts):
            vectors = [list(v) for _, v in upserts]
            if vectors and float(vectors[0][0]) == 1.0:
                time.sleep(0.2)
            with self._lock:
                for i in delete_ids:
                    self.by_id.pop(int(i), None)
                for doc_id, vector in upserts:
                    self.by_id[int(doc_id)] = list(vector)

    fake_vec = FakeVec()
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "SqlDocumentStorage", FakeRepo, raising=True)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: FakeEmbedder(), raising=True)
    monkeypatch.setattr(factory, "VectorStorage", lambda *a, **k: fake_vec, raising=True)
    monkeypatch.setattr(docs_router, "reset_rag_service", lambda: None, raising=True)

    task_a = asyncio.create_task(
        asgi_client.post(
            "/api/docs/mutate",
            json={"upserts": [{"external_id": "doc-1", "content": "A"}]},
        )
    )
    await asyncio.sleep(0.02)
    task_b = asyncio.create_task(
        asgi_client.post(
            "/api/docs/mutate",
            json={"upserts": [{"external_id": "doc-1", "content": "B"}]},
        )
    )
    ra, rb = await asyncio.gather(task_a, task_b)

    assert ra.status_code == 200
    assert rb.status_code == 200
    assert FakeRepo._by_external_id["doc-1"][1] == "B"
    assert fake_vec.by_id[1] == [2.0]


async def test_docs_ingest_executes_single_locked_mutation_pass(
    asgi_client, in_memory_sqlite, monkeypatch
):
    called_funcs: list[str] = []
    called_task_types: list[str] = []

    async def _fake_run_blocking(func, /, *args, **kwargs):
        called_task_types.append(str(kwargs.pop("task_type", "default")))
        called_funcs.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(docs_router, "run_blocking", _fake_run_blocking, raising=True)
    monkeypatch.setattr(docs_router, "reset_rag_service", lambda: None, raising=True)

    resp = await asgi_client.post("/api/docs", json={"texts": ["  hello world  "]})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] >= 1
    assert called_funcs.count("run_multi_store_write_locked") == 0
    assert len(called_funcs) == 1
    assert called_funcs[0] != "run_multi_store_write_locked"
    assert called_task_types == ["mutation"]
    assert "_ingest_sync" not in called_funcs


async def test_mutation_failure_still_invalidates_cached_rag_service(
    asgi_client, in_memory_sqlite, monkeypatch
):
    class FakeEmbedder:
        dim = 1

        def embed(self, texts):
            return [[1.0] for _ in texts]

    class FailingVec:
        def apply_delta_atomic(self, *, delete_ids, upserts):
            raise RuntimeError("vec upsert failed")

    reset_calls = 0

    def _count_reset() -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: FakeEmbedder(), raising=True)
    monkeypatch.setattr(factory, "VectorStorage", lambda *a, **k: FailingVec(), raising=True)
    monkeypatch.setattr(docs_router, "reset_rag_service", _count_reset, raising=True)

    with pytest.raises(RuntimeError, match="vec upsert failed"):
        await asgi_client.post(
            "/api/docs/mutate",
            json={"upserts": [{"external_id": "doc-1", "content": "hello"}]},
        )

    assert reset_calls == 1
    assert SqlDocumentStorage().get_all_documents() == []
