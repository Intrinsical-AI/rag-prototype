from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage


def test_faiss_index_delete_ids_removes_vectors(tmp_path):
    """
    Validate that vector deletions actually take effect (numpy fallback path is enough).
    This is important both for ETL rollback and for user-initiated purges.
    """
    idx_path = tmp_path / "idx.npy"
    map_path = tmp_path / "id_map.json"

    st = FaissVectorStorage(index_path=str(idx_path), id_map_path=str(map_path), dim=2)
    st.upsert([1, 2, 3], [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])

    # Delete the "closest" vector to the query; ensure it no longer appears.
    st.delete([2])
    pairs = st.similar([10.0, 0.0], k=3)
    ids = [doc_id for doc_id, _ in pairs]
    assert 2 not in ids

    # Internal id_map persisted should not contain the deleted ID.
    persisted = map_path.read_text(encoding="utf-8")
    assert "2" not in persisted


def test_delete_is_idempotent(tmp_path):
    idx_path = tmp_path / "idx.npy"
    map_path = tmp_path / "id_map.json"

    st = FaissVectorStorage(index_path=str(idx_path), id_map_path=str(map_path), dim=2)
    st.upsert([1], [[0.0, 0.0]])
    st.delete([999])  # no-op
    st.delete([1])
    st.delete([1])  # repeat no-op

    assert st.similar([0.0, 0.0], k=1) == []
