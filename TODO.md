    # Optional FAISS index generation for dense retrieval
    if settings.retrieval_mode == "dense":
        embedding_dim = 384  # Fixed dimension for testing/demo purposes
        rng = np.random.default_rng(seed=42)
        vectors = rng.random((len(texts), embedding_dim), dtype=np.float32)
        _build_dense_index(vectors, ids)
