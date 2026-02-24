from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest


def _lock_worker(lock_dir: str, active: object, max_active: object) -> None:
    # Import inside worker so spawned processes initialize cleanly.
    from local_rag_backend.core.services.write_lock import multi_store_write_lock

    active_v = active
    max_active_v = max_active
    with multi_store_write_lock(coordination_dir=Path(lock_dir)):
        with active_v.get_lock():
            active_v.value += 1
            if active_v.value > max_active_v.value:
                max_active_v.value = active_v.value
        time.sleep(0.25)
        with active_v.get_lock():
            active_v.value -= 1


@pytest.mark.integration
def test_multi_store_write_lock_serializes_across_processes(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    active = ctx.Value("i", 0)
    max_active = ctx.Value("i", 0)
    lock_dir = tmp_path / "coord"
    lock_dir.mkdir(parents=True, exist_ok=True)

    p1 = ctx.Process(target=_lock_worker, args=(str(lock_dir), active, max_active))
    p2 = ctx.Process(target=_lock_worker, args=(str(lock_dir), active, max_active))
    p1.start()
    p2.start()

    p1.join(timeout=10)
    p2.join(timeout=10)
    if p1.is_alive():
        p1.terminate()
    if p2.is_alive():
        p2.terminate()

    assert p1.exitcode == 0
    assert p2.exitcode == 0
    assert max_active.value == 1
