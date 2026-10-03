from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

from local_rag_backend.composition import factory
from local_rag_backend.core.use_cases.docs_mutation import MutationCoordinator, MutationIntent
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any

    from local_rag_backend.core.use_cases.results import MutationSummary


async def test_canonical_outer_lock_and_concurrent_mutation_both_finish(
    asgi_client, in_memory_sqlite, monkeypatch
) -> None:
    """A waiting mutation must not become a leader that blocks the import's nested write."""
    monkeypatch.setattr(get_settings(), "retrieval_mode", "sparse")
    container = factory.get_app_context().container
    original_write_lock = container.write_lock
    original_run_locked = container.run_multi_store_write_locked
    original_execute = MutationCoordinator.execute
    outer_acquired = threading.Event()
    competing_lock_requested = threading.Event()
    outer_thread: list[int] = []
    competing_thread: list[int] = []

    def observed_execute(self: MutationCoordinator, intent: MutationIntent) -> MutationSummary:
        if intent.source == "api:/docs/mutate" and any(
            item.external_id == "concurrent" for item in intent.upserts
        ):
            competing_thread.append(threading.get_ident())
        return original_execute(self, intent)

    @contextmanager
    def observed_write_lock(*args: Any, **kwargs: Any) -> Iterator[None]:
        if (
            outer_thread
            and competing_thread
            and threading.get_ident() == competing_thread[0] != outer_thread[0]
        ):
            competing_lock_requested.set()
        with original_write_lock(*args, **kwargs):
            yield

    def gated_import(fn: Callable[[], Any]) -> Any:
        def after_outer_lock() -> Any:
            outer_thread.append(threading.get_ident())
            outer_acquired.set()
            assert competing_lock_requested.wait(timeout=3), (
                "concurrent mutation never attempted the write lock"
            )
            return fn()

        return original_run_locked(after_outer_lock)

    monkeypatch.setattr(container, "write_lock", observed_write_lock)
    monkeypatch.setattr(container, "run_multi_store_write_locked", gated_import)
    monkeypatch.setattr(MutationCoordinator, "execute", observed_execute)

    import_task = asyncio.create_task(
        asgi_client.post(
            "/api/docs/import-canonical",
            json={
                "scope": "concurrency-test",
                "snapshot_id": "one",
                "replace_scope": False,
                "documents": [{"external_id": "canonical", "content": "canonical text"}],
            },
        )
    )
    assert await asyncio.to_thread(outer_acquired.wait, 3), "canonical import never held outer lock"

    mutation_task = asyncio.create_task(
        asgi_client.post(
            "/api/docs/mutate",
            json={"upserts": [{"external_id": "concurrent", "content": "concurrent text"}]},
        )
    )
    imported, mutated = await asyncio.wait_for(
        asyncio.gather(import_task, mutation_task), timeout=5
    )
    assert imported.status_code == 200, imported.text
    assert mutated.status_code == 200, mutated.text

    later = await asyncio.wait_for(
        asgi_client.post(
            "/api/docs/mutate",
            json={"upserts": [{"external_id": "later", "content": "later text"}]},
        ),
        timeout=5,
    )
    assert later.status_code == 200, later.text
    assert {
        doc.external_id for doc in SqlDocumentStorage(in_memory_sqlite).get_all_documents()
    } == {"canonical", "concurrent", "later"}
