"""In-process mutation batching primitives."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Condition, Event, Lock
from time import monotonic
from typing import Any, ClassVar


@dataclass
class MutationBatchItem:
    payload: Any
    done: Event = field(default_factory=Event)
    result: Any = None
    error: Exception | None = None


@dataclass
class _BatchState:
    condition: Condition = field(default_factory=Condition)
    queue: deque[MutationBatchItem] = field(default_factory=deque)
    draining: bool = False


class MutationBatchCoordinator:
    """Coalesces concurrent requests into bounded batches per queue key."""

    _STATES_LOCK: ClassVar[Lock] = Lock()
    _STATES: ClassVar[dict[str, _BatchState]] = {}

    def submit(
        self,
        *,
        queue_key: str,
        payload: Any,
        max_batch_size: int,
        max_wait_ms: int,
        process_batch: Callable[[list[MutationBatchItem]], None],
    ) -> Any:
        item = MutationBatchItem(payload=payload)
        state = self._get_state(queue_key)

        is_leader = False
        with state.condition:
            state.queue.append(item)
            state.condition.notify_all()
            if not state.draining:
                state.draining = True
                is_leader = True

        if is_leader:
            self._drain_loop(
                state=state,
                max_batch_size=max_batch_size,
                max_wait_ms=max_wait_ms,
                process_batch=process_batch,
            )

        item.done.wait()
        if item.error is not None:
            raise item.error
        return item.result

    def _get_state(self, queue_key: str) -> _BatchState:
        with self._STATES_LOCK:
            state = self._STATES.get(queue_key)
            if state is None:
                state = _BatchState()
                self._STATES[queue_key] = state
            return state

    def _take_next_batch(
        self,
        *,
        state: _BatchState,
        max_batch_size: int,
        max_wait_ms: int,
    ) -> list[MutationBatchItem]:
        size = max(1, min(int(max_batch_size), 512))
        wait_s = max(0.0, min(int(max_wait_ms), 5000) / 1000.0)

        with state.condition:
            if not state.queue:
                return []
            if wait_s > 0.0 and len(state.queue) < size:
                deadline = monotonic() + wait_s
                while len(state.queue) < size:
                    remaining = deadline - monotonic()
                    if remaining <= 0.0:
                        break
                    state.condition.wait(timeout=remaining)
            n = min(len(state.queue), size)
            return [state.queue.popleft() for _ in range(n)]

    def _drain_loop(
        self,
        *,
        state: _BatchState,
        max_batch_size: int,
        max_wait_ms: int,
        process_batch: Callable[[list[MutationBatchItem]], None],
    ) -> None:
        while True:
            batch = self._take_next_batch(
                state=state,
                max_batch_size=max_batch_size,
                max_wait_ms=max_wait_ms,
            )
            if batch:
                try:
                    process_batch(batch)
                except Exception as exc:
                    for item in batch:
                        if not item.done.is_set():
                            item.error = exc
                            item.done.set()
                continue

            with state.condition:
                if state.queue:
                    continue
                state.draining = False
                return


__all__ = ["MutationBatchCoordinator", "MutationBatchItem"]
