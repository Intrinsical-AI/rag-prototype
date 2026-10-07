from __future__ import annotations

import asyncio
from threading import Event

import pytest

from local_rag_backend.infrastructure.concurrency.blocking import BlockingExecutor
from local_rag_backend.settings import Settings


async def _wait_started(started: Event) -> None:
    async with asyncio.timeout(2):
        while not started.is_set():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_running_cancellation_retains_admission_until_worker_finishes(monkeypatch):
    executor = BlockingExecutor(
        settings_obj=Settings(blocking_workers_mutation=1, blocking_queue_mutation=1)
    )
    state = executor._get_executor_state("mutation")
    started, finish = Event(), Event()
    releases = []
    release = state.release_slot

    def counted_release():
        releases.append(state.pending_snapshot())
        release()

    monkeypatch.setattr(state, "release_slot", counted_release)

    def worker():
        started.set()
        assert finish.wait(3)

    first = asyncio.create_task(executor.run_blocking(worker, task_type="mutation"))
    second = None
    try:
        await _wait_started(started)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert releases == []
        assert state.pending_snapshot() == 1
        second = asyncio.create_task(executor.run_blocking(lambda: 2, task_type="mutation"))
        await asyncio.sleep(0.01)
        assert state.pending_snapshot() == 2
        with pytest.raises(RuntimeError, match="queue is full"):
            await executor.run_blocking(lambda: 3, task_type="mutation")
        finish.set()
        assert await second == 2
        assert len(releases) == 2
        assert state.pending_snapshot() == 0
    finally:
        finish.set()
        if second is not None:
            await second
        executor.close()


@pytest.mark.asyncio
async def test_queued_cancellation_releases_once_without_running_worker(monkeypatch):
    executor = BlockingExecutor(
        settings_obj=Settings(blocking_workers_mutation=1, blocking_queue_mutation=1)
    )
    state = executor._get_executor_state("mutation")
    started, finish, queued_started = Event(), Event(), Event()
    releases = []
    release = state.release_slot

    def counted_release():
        releases.append(state.pending_snapshot())
        release()

    monkeypatch.setattr(state, "release_slot", counted_release)

    def worker():
        started.set()
        assert finish.wait(3)

    first = asyncio.create_task(executor.run_blocking(worker, task_type="mutation"))
    try:
        await _wait_started(started)
        queued = asyncio.create_task(
            executor.run_blocking(queued_started.set, task_type="mutation")
        )
        await asyncio.sleep(0.01)
        assert state.pending_snapshot() == 2
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert len(releases) == 1
        assert state.pending_snapshot() == 1
        assert not queued_started.is_set()
        finish.set()
        await first
        assert len(releases) == 2
        assert state.pending_snapshot() == 0
    finally:
        finish.set()
        await first
        executor.close()


@pytest.mark.asyncio
async def test_submission_failure_releases_admitted_capacity(monkeypatch):
    executor = BlockingExecutor(settings_obj=Settings())
    state = executor._get_executor_state("mutation")

    def submit(_call):
        raise RuntimeError("submission failed")

    monkeypatch.setattr(state.executor, "submit", submit)
    try:
        with pytest.raises(RuntimeError, match="submission failed"):
            await executor.run_blocking(lambda: 1, task_type="mutation")
        assert state.pending_snapshot() == 0
    finally:
        executor.close()
