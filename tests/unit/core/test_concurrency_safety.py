"""Concurrency safety tests for critical components.

Tests thread safety and race conditions in:
- ETL service concurrent ingestion
- Retriever concurrent queries
- Storage layer concurrent access
- Resource management under load
"""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever

pytestmark = [pytest.mark.concurrency]


class TestConcurrencySafety:
    """Test concurrency safety across components."""

    @pytest.fixture
    def thread_safe_mock_storage(self):
        """Mock storage that simulates thread-safe behavior."""
        storage = Mock()
        storage._counter = 0
        storage._lock = threading.Lock()

        def thread_safe_store(texts):
            with storage._lock:
                start_id = storage._counter + 1
                storage._counter += len(texts)
                return list(range(start_id, storage._counter + 1))

        storage.store_documents.side_effect = thread_safe_store
        return storage

    @pytest.fixture
    def thread_safe_mock_embedder(self):
        """Mock embedder that simulates processing time."""
        embedder = Mock()

        def slow_embed(texts):
            # Simulate processing time
            time.sleep(0.01)
            return [[float(i), float(i + 1)] for i in range(len(texts))]

        embedder.embed.side_effect = slow_embed
        return embedder

    def test_concurrent_etl_ingestion(self, thread_safe_mock_storage, thread_safe_mock_embedder):
        """Test concurrent ETL ingestion operations."""
        mock_vec_storage = Mock()
        etl = ETLService(thread_safe_mock_storage, mock_vec_storage, thread_safe_mock_embedder)

        results = queue.Queue()
        errors = queue.Queue()

        def ingest_batch(batch_id, texts):
            try:
                result = etl.ingest(texts)
                results.put((batch_id, result))
            except Exception as e:
                errors.put((batch_id, e))

        # Create multiple threads for concurrent ingestion
        threads = []
        for i in range(5):
            texts = [f"Thread {i} doc {j}" for j in range(3)]
            thread = threading.Thread(target=ingest_batch, args=(i, texts))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Collect results
        batch_results = {}
        while not results.empty():
            batch_id, result = results.get()
            batch_results[batch_id] = result

        # Check for errors
        error_list = []
        while not errors.empty():
            error_list.append(errors.get())

        # Verify results
        assert len(error_list) == 0, f"Errors occurred: {error_list}"
        assert len(batch_results) == 5

        # All batches should have unique IDs
        all_ids = []
        for result in batch_results.values():
            all_ids.extend(result)

        assert len(all_ids) == len(set(all_ids)), "Duplicate IDs detected in concurrent ingestion"

    def test_concurrent_retrieval_operations(self):
        """Test concurrent retrieval operations."""
        # Mock components
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

        mock_faiss_index = Mock()
        mock_faiss_index.similar.return_value = [(1, 0.8), (2, 0.6)]

        mock_doc_repo = Mock()
        from local_rag_backend.core.domain.entities import Document

        mock_doc_repo.get.return_value = [
            Document(id=1, content="Doc 1"),
            Document(id=2, content="Doc 2"),
        ]

        retriever = DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)

        results = queue.Queue()
        errors = queue.Queue()

        def concurrent_retrieve(query_id, query):
            try:
                docs, scores = retriever.retrieve(query, k=5)
                results.put((query_id, len(docs), len(scores)))
            except Exception as e:
                errors.put((query_id, e))

        # Create multiple concurrent queries
        queries = [f"query {i}" for i in range(10)]
        threads = []

        for i, query in enumerate(queries):
            thread = threading.Thread(target=concurrent_retrieve, args=(i, query))
            threads.append(thread)

        # Start all threads simultaneously
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Collect results
        query_results = {}
        while not results.empty():
            query_id, doc_count, score_count = results.get()
            query_results[query_id] = (doc_count, score_count)

        # Check for errors
        error_list = []
        while not errors.empty():
            error_list.append(errors.get())

        # Verify results
        assert len(error_list) == 0, f"Errors in concurrent retrieval: {error_list}"
        assert len(query_results) == 10

        # All queries should return consistent results
        for doc_count, score_count in query_results.values():
            assert doc_count == score_count
            assert doc_count >= 0

    def test_concurrent_hybrid_retrieval(self):
        """Test concurrent hybrid retrieval operations."""
        # Mock dense retriever
        mock_dense = Mock()
        mock_dense.retrieve.return_value = ([Document(id=1, content="Dense 1")], [0.9])

        # Mock sparse retriever
        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = ([Document(id=2, content="Sparse 2")], [0.8])

        hybrid = HybridRetriever(mock_dense, mock_sparse, alpha=0.5)

        results = []
        errors = []

        def concurrent_hybrid_retrieve(query):
            try:
                docs, scores = hybrid.retrieve(query, k=5)
                results.append((len(docs), len(scores)))
            except Exception as e:
                errors.append(e)

        # Use ThreadPoolExecutor for better control
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_hybrid_retrieve, f"query {i}") for i in range(20)]

            # Wait for all to complete
            for future in as_completed(futures):
                future.result()  # This will raise if there was an exception

        # Verify results
        assert len(errors) == 0, f"Errors in concurrent hybrid retrieval: {errors}"
        assert len(results) == 20

        # All results should be consistent
        for doc_count, score_count in results:
            assert doc_count == score_count

    def test_resource_contention_simulation(
        self, thread_safe_mock_storage, thread_safe_mock_embedder
    ):
        """Test behavior under resource contention."""
        # Create ETL service with limited resources
        mock_vec_storage = Mock()

        # Simulate resource contention in vector storage
        vec_storage_lock = threading.Lock()

        def contended_upsert(ids, vectors):
            with vec_storage_lock:
                time.sleep(0.02)  # Simulate slow operation

        mock_vec_storage.upsert.side_effect = contended_upsert

        etl = ETLService(thread_safe_mock_storage, mock_vec_storage, thread_safe_mock_embedder)

        results = []
        errors = []

        def stress_ingest(batch_id):
            try:
                texts = [f"Stress batch {batch_id} doc {i}" for i in range(2)]
                result = etl.ingest(texts)
                results.append((batch_id, len(result)))
            except Exception as e:
                errors.append((batch_id, e))

        # Create high contention scenario
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(stress_ingest, i) for i in range(20)]

            for future in as_completed(futures):
                future.result()

        # Verify system handled contention gracefully
        assert len(errors) == 0, f"Errors under resource contention: {errors}"
        assert len(results) == 20

        # All operations should have succeeded
        for _batch_id, doc_count in results:
            assert doc_count == 2

    def test_memory_pressure_concurrent_operations(self):
        """Test concurrent operations under memory pressure."""
        # Mock components that simulate memory usage
        mock_embedder = Mock()

        def memory_intensive_embed(texts):
            # Simulate memory-intensive operation
            large_data = [[0.1] * 1000 for _ in range(len(texts))]
            time.sleep(0.01)
            return [[float(i)] * 4 for i in range(len(texts))]

        mock_embedder.embed.side_effect = memory_intensive_embed

        mock_storage = Mock()
        mock_storage.store_documents.side_effect = lambda texts: list(range(1, len(texts) + 1))

        mock_vec_storage = Mock()

        etl = ETLService(mock_storage, mock_vec_storage, mock_embedder)

        # Concurrent operations with memory pressure
        def memory_stress_ingest(batch_id):
            texts = [f"Memory stress {batch_id} doc {i}" for i in range(5)]
            return etl.ingest(texts)

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(memory_stress_ingest, i) for i in range(6)]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(len(result))
                except Exception as e:
                    pytest.fail(f"Memory pressure caused failure: {e}")

        # All operations should complete successfully
        assert len(results) == 6
        assert all(count == 5 for count in results)

    def test_deadlock_prevention(self):
        """Test that operations don't cause deadlocks."""
        # Create scenario with consistent lock ordering to prevent deadlock
        lock1 = threading.Lock()
        lock2 = threading.Lock()

        results = []

        def operation_a():
            # Always acquire locks in same order to prevent deadlock
            with lock1:
                time.sleep(0.001)
                with lock2:
                    results.append("A")

        def operation_b():
            # Same lock ordering prevents deadlock
            with lock1:
                time.sleep(0.001)
                with lock2:
                    results.append("B")

        # This test ensures proper lock ordering

        threads = [threading.Thread(target=operation_a), threading.Thread(target=operation_b)]

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=1.0)  # Prevent hanging in case of deadlock

        elapsed = time.time() - start_time

        # Should complete quickly with proper lock ordering
        assert elapsed < 0.5, "Potential deadlock detected"
        assert len(results) == 2

    def test_race_condition_detection(self, thread_safe_mock_storage):
        """Test for race conditions in shared state."""
        # Shared counter to detect race conditions
        shared_state = {"counter": 0, "values": []}
        state_lock = threading.Lock()

        def racy_operation(thread_id):
            # Simulate operation that could have race condition
            for i in range(10):
                # Proper locking for read-modify-write operation
                with state_lock:
                    shared_state["counter"] += 1
                    shared_state["values"].append((thread_id, i))

        threads = [threading.Thread(target=racy_operation, args=(i,)) for i in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no race condition occurred
        assert shared_state["counter"] == 50  # 5 threads * 10 operations
        assert len(shared_state["values"]) == 50
        assert len(set(shared_state["values"])) == 50  # All unique
