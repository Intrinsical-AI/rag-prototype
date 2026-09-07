from __future__ import annotations

import io
import json

from local_rag_backend.core.use_cases import docs_ingest
from local_rag_backend.core.use_cases.results import MutationSummary, UpsertDocResult
from local_rag_backend.http.routers import docs as docs_router
from local_rag_backend.settings import settings


async def test_docs_mutate_endpoint_calls_coordinator_execute_once(
    asgi_client, in_memory_sqlite, monkeypatch
):
    calls = 0

    class FakeCoordinator:
        def __init__(self, *, settings_obj, ports):
            pass

        def execute(self, intent):
            nonlocal calls
            calls += 1
            return MutationSummary(
                op_id=str(intent.op_id or "fake-op"),
                inserted=1,
                results=[
                    UpsertDocResult(
                        external_id="doc-1",
                        id="1",
                        action="inserted",
                        content_changed=True,
                    )
                ],
            )

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(docs_router, "MutationCoordinator", FakeCoordinator, raising=True)

    resp = await asgi_client.post(
        "/api/docs/mutate",
        json={"upserts": [{"external_id": "doc-1", "content": "hello"}]},
    )
    assert resp.status_code == 200
    assert calls == 1


async def test_docs_ingest_and_import_endpoints_call_coordinator_execute(
    asgi_client, in_memory_sqlite, monkeypatch
):
    calls = 0

    class FakeCoordinator:
        def __init__(self, *, settings_obj, ports):
            pass

        def execute(self, intent):
            nonlocal calls
            calls += 1
            results = [
                UpsertDocResult(
                    external_id=str(item.external_id),
                    id=f"id-{i}",
                    action="inserted",
                    content_changed=True,
                )
                for i, item in enumerate(intent.upserts, start=1)
            ]
            return MutationSummary(
                op_id=str(intent.op_id or "fake-op"),
                inserted=len(results),
                results=results,
            )

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(docs_ingest, "MutationCoordinator", FakeCoordinator, raising=True)

    ingest_resp = await asgi_client.post("/api/docs/ingest", json={"texts": ["hello"]})
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["count"] >= 1

    export = [
        {
            "title": "Test Conversation",
            "conversation_id": "conv-123",
            "mapping": {
                "msg-1": {
                    "id": "msg-1",
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "create_time": 1700000000.0,
                        "content": {"content_type": "text", "parts": ["Hello import"]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                }
            },
        }
    ]
    files = {
        "file": (
            "export.json",
            io.BytesIO(json.dumps(export).encode()),
            "application/json",
        )
    }
    import_resp = await asgi_client.post("/api/docs/import-conversations", files=files)
    assert import_resp.status_code == 200
    assert import_resp.json()["count"] >= 1

    assert calls == 2
