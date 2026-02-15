import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import numpy as np

from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex


def test_search_during_add_does_not_raise_or_mis_map_ids(tmp_path):
    """
    Regression test for a race where the index vectors were updated before `id_map`,
    causing callers to map an index position that didn't exist yet (IndexError / wrong IDs).
    """

    index_path = tmp_path / "i.faiss"
    id_map_path = tmp_path / "m.json"
    fi = FaissIndex(index_path, id_map_path, dim=4)

    extend_called = Event()
    release_extend = Event()

    class _BlockingList(list):
        def extend(self, it):
            extend_called.set()
            release_extend.wait(timeout=2)
            return super().extend(it)

    # Block the id-map update mid-write to simulate an unlucky interleaving.
    # Note: `add_to_index()` reloads on-disk state and reassigns `self.id_map`, so we
    # patch the loader to force our blocking list to be used after reload as well.
    blocking_list = _BlockingList()
    fi.id_map = blocking_list
    fi._load_id_map_locked = lambda: blocking_list  # type: ignore[method-assign]

    vec = np.zeros(4, dtype="float32")

    t = Thread(target=lambda: fi.add_to_index([123], [vec]))
    t.start()

    try:
        assert extend_called.wait(timeout=2), "add_to_index did not reach id_map.extend()"

        def _search_and_map():
            idxs, _dists = fi.search(vec, k=1)
            return fi.id_map[idxs[0]]

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_search_and_map)
            # Ensure the search had a chance to start and would fail without locking.
            time.sleep(0.05)
            release_extend.set()
            assert fut.result(timeout=2) == 123
    finally:
        release_extend.set()
        t.join(timeout=2)
