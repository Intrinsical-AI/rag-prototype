# src/infrastructure/persistence/faiss/faiss_.py

from collections.abc import Sequence

from local_rag_backend.core.ports import VectorRepoPort
from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex


class FaissVectorStorage(VectorRepoPort):
    """
    Adapter que implementa VectorRepoPort usando FAISS.
    """

    def __init__(self, index_path: str, id_map_path: str, dim: int | None = None):
        self.faiss_index = FaissIndex(index_path, id_map_path, dim=dim or 384)

    def upsert(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        self.faiss_index.add_to_index(list(ids), list(vectors))

    def similar(self, vector, k: int):
        idxs, dists = self.faiss_index.search(vector, k)
        # Convert L2 distances to similarities and normalize to [0,1]
        # Filter out invalid entries (-1) before normalization
        valid_pairs = [(i, d) for i, d in zip(idxs, dists) if i != -1]
        if valid_pairs:
            valid_dists = [d for _, d in valid_pairs]
            sims_raw = [1.0 / (1.0 + float(d)) for d in valid_dists]
            min_s, max_s = min(sims_raw), max(sims_raw)
            if max_s == min_s:
                sims = [0.0 if max_s == 0 else 1.0] * len(sims_raw)
            else:
                sims = [(s - min_s) / (max_s - min_s) for s in sims_raw]
        else:
            sims = []
        pairs: list[tuple[int, float]] = []
        for (i, _), sim in zip(valid_pairs, sims, strict=False):
            real_id = self.faiss_index.id_map[i]
            pairs.append((real_id, float(sim)))
        return pairs


"""
faiss = DenseFaissRetriever(embedder=embedder, doc_repo=sql_repo, ...)
retriever = IdMapperRetriever(faiss, sql_repo)
"""
