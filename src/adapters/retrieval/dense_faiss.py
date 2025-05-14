"""
File: src/adapters/retrieval/dense_faiss.py
FAISS dense retrieval. Optional / heavy.
"""

import faiss
import pickle
import numpy as np
from typing import List, Tuple
from src.settings import settings
from src.core.ports import RetrieverPort, EmbedderPort

class DenseFaissRetriever(RetrieverPort):
    def __init__(self, embedder: EmbedderPort):
        self.index = faiss.read_index(settings.index_path)
        with open(settings.id_map_path, "rb") as fh:
            self.id_map = pickle.load(fh)
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        vec = np.array(self.embedder.embed(query), dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(vec, k)
        # ids   = [self.id_map[i] for i in I[0]]
        ids = [self.id_map[idx] for idx in indices[0] if idx != -1]
        # dists = D[0].tolist()
        dists = distances[0].tolist()
        # Convert distance to similarity (better for retrieval) --> simil = 1 / (1 + dist) (the bigger the better)
        sims  = [1.0 / (1.0 + d) for d in dists]
        # Normalize between [0,1]
        if sims:
            mx, mn = max(sims), min(sims)
            if abs(mn-mx) > 0.0001:
                norm = [(s - mn) / (mx - mn + 1e-9) for s in sims]
            else:
                norm = [0.0]*len(sims)
        else:
            norm = []
        return ids, norm
