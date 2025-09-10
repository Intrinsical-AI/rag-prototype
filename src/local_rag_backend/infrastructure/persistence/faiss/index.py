import pickle
from collections.abc import Sequence
from pathlib import Path

import faiss  # type: ignore
import numpy as np


class FaissIndex:
    def __init__(self, index_path, id_map_path, dim=384):
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.dim = dim
        self._load()

    def _load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatL2(self.dim)
        if self.id_map_path.exists():
            with self.id_map_path.open("rb") as f:
                self.id_map = pickle.load(f)
        else:
            self.id_map = []

    def add_to_index(self, ids: list[int], embeddings: list[Sequence[float]]):
        vectors = np.asarray(embeddings, dtype="float32")
        # Validate dimensionality matches index.d to avoid FAISS errors later
        if vectors.ndim != 2 or vectors.shape[1] != self.index.d:
            raise ValueError(
                f"FAISS dim mismatch: index {self.index.d} vs vectors {vectors.shape}"
            )
        self.index.add(vectors)
        self.id_map.extend(ids)
        self.save()

    def search(self, query_vector: Sequence[float], k: int):
        vectors = np.asarray([query_vector], dtype="float32")
        scores, idxs = self.index.search(vectors, k)
        return idxs[0], scores[0]

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with self.id_map_path.open("wb") as f:
            pickle.dump(self.id_map, f)
