from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from dataclasses import dataclass

import pytest

from local_rag_backend.app import factory
from local_rag_backend.app.routers import docs as docs_router
from local_rag_backend.settings import settings


async def test_concurrent_upserts_are_serialized_and_keep_sql_vector_consistent(
    asgi_client, in_memory_sqlite, monkeypatch
):
    @dataclass(frozen=True)
    class _UpsertDoc:
        external_id: str
        content: str
        source_id: str | None = None
        metadata: dict[str, object] | None = None
        chunk_dedup_sha256: str | None = None

    @dataclass(frozen=True)
    class _UpsertResult:
        external_id: str
        id: int
        action: str
        content_changed: bool

    @dataclass(frozen=True)
    class _ExistingDocState:
        id: int
        external_id: str
        content: str
        content_sha256: str | None

    class FakeRepo:
        UpsertDoc = _UpsertDoc

        _lock = threading.Lock()
        _next_id = 1
        _by_external_id: dict[str, tuple[int, str, str]] = {}
        _a_sql_done = threading.Event()

        def get_tombstoned_external_ids(self, external_ids):
            return set()

        def get_existing_doc_states_by_external_id(self, external_ids):
            with self._lock:
                out: dict[str, _ExistingDocState] = {}
                for ext in external_ids:
                    row = self._by_external_id.get(ext)
                    if row is None:
                        continue
                    doc_id, content, sha = row
                    out[ext] = _ExistingDocState(
                        id=doc_id, external_id=ext, content=content, content_sha256=sha
                    )
                return out

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
                        # Forces the B request to update SQL while A still sleeps in vector upsert.
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

        def delete(self, ids):
            with self._lock:
                for i in ids:
                    self.by_id.pop(int(i), None)

        def upsert(self, ids, vectors):
            # Force an interleaving window where request A sleeps after SQL commit.
            if vectors and float(vectors[0][0]) == 1.0:
                time.sleep(0.2)
            with self._lock:
                for i, v in zip(ids, vectors, strict=False):
                    self.by_id[int(i)] = list(v)

    fake_vec = FakeVec()
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "SqlDocumentStorage", FakeRepo, raising=True)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: FakeEmbedder(), raising=True)
    monkeypatch.setattr(factory, "FaissVectorStorage", lambda *a, **k: fake_vec, raising=True)
    monkeypatch.setattr(docs_router, "reset_rag_service", lambda: None, raising=True)

    task_a = asyncio.create_task(
        asgi_client.post(
            "/api/docs/upsert",
            json={"docs": [{"external_id": "doc-1", "content": "A"}]},
        )
    )
    await asyncio.sleep(0.02)
    task_b = asyncio.create_task(
        asgi_client.post(
            "/api/docs/upsert",
            json={"docs": [{"external_id": "doc-1", "content": "B"}]},
        )
    )
    ra, rb = await asyncio.gather(task_a, task_b)

    assert ra.status_code == 200
    assert rb.status_code == 200
    # Final SQL state must match final vector state (latest content = B -> vector 2.0).
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

    # Regression guard: /api/docs must run exactly one sync ingestion path under the lock wrapper.
    assert called_funcs.count("run_multi_store_write_locked") == 1
    assert called_task_types == ["mutation"]
    assert "_ingest_sync" not in called_funcs


async def test_upsert_failure_still_invalidates_cached_rag_service(
    asgi_client, in_memory_sqlite, monkeypatch
):
    @dataclass(frozen=True)
    class _UpsertDoc:
        external_id: str
        content: str
        source_id: str | None = None
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

        def get_tombstoned_external_ids(self, external_ids):
            return set()

        def get_existing_doc_states_by_external_id(self, external_ids):
            return {}

        def upsert_documents_by_external_id(self, items):
            return (
                [
                    _UpsertResult(
                        external_id=items[0].external_id,
                        id=1,
                        action="inserted",
                        content_changed=True,
                    )
                ],
                [(1, items[0].content)],
                [],
            )

    class FakeEmbedder:
        dim = 1

        def embed(self, texts):
            return [[1.0] for _ in texts]

    class FailingVec:
        def delete(self, ids):
            return None

        def upsert(self, ids, vectors):
            raise RuntimeError("vec upsert failed")

    reset_calls = 0

    def _count_reset() -> None:
        nonlocal reset_calls
        reset_calls += 1

    def _rebuild_fail(**_kwargs):
        raise RuntimeError("rebuild failed")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "SqlDocumentStorage", FakeRepo, raising=True)
    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: FakeEmbedder(), raising=True)
    monkeypatch.setattr(factory, "FaissVectorStorage", lambda *a, **k: FailingVec(), raising=True)
    monkeypatch.setattr(factory, "rebuild_index_from_db", _rebuild_fail, raising=True)
    monkeypatch.setattr(docs_router, "reset_rag_service", _count_reset, raising=True)

    with pytest.raises(RuntimeError, match="rebuild failed"):
        await asgi_client.post(
            "/api/docs/upsert",
            json={"docs": [{"external_id": "doc-1", "content": "hello"}]},
        )

    assert reset_calls == 1
