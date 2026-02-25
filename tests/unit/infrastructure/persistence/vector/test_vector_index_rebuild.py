from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage


def test_rebuild_is_idempotent(tmp_path):
    idx_path = tmp_path / "idx.npy"
    map_path = tmp_path / "id_map.json"

    st = VectorStorage(index_path=str(idx_path), id_map_path=str(map_path), dim=2)
    st.rebuild(["doc:1", "doc:2"], [[0.0, 0.0], [10.0, 0.0]])
    first = st.vector_index.id_map[:]

    # Rebuild again with the same content should not duplicate.
    st.rebuild(["doc:1", "doc:2"], [[0.0, 0.0], [10.0, 0.0]])
    second = st.vector_index.id_map[:]

    assert first == ["doc:1", "doc:2"]
    assert second == ["doc:1", "doc:2"]


def test_rebuild_replaces_previous_state(tmp_path):
    idx_path = tmp_path / "idx.npy"
    map_path = tmp_path / "id_map.json"

    st = VectorStorage(index_path=str(idx_path), id_map_path=str(map_path), dim=2)
    st.upsert(["doc:1", "doc:2", "doc:3"], [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])

    st.rebuild(["doc:2"], [[10.0, 0.0]])
    assert st.vector_index.id_map == ["doc:2"]
