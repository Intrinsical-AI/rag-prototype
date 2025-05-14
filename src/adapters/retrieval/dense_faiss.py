from __future__ import annotations

"""Dense FAISS retriever

Refactor 2025‑05‑15
-------------------

* Acepta un `embedder` inyectable para evitar descargas en tests.
* Tolera ambos formatos de `id_map` (lista o dict) para back‑compat.
* Comprueba la consistencia de la dimensión entre índice y embedder.
* API pública: ``retrieve(query: str, k: int = 5)`` → (ids, scores).
"""

from pathlib import Path
import pickle
from typing import List, Tuple, Sequence

import faiss  # type: ignore
import numpy as np

from src.core.ports import RetrieverPort, EmbedderPort
from src.settings import settings
from src.adapters.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)

__all__ = ["DenseFaissRetriever"]


class DenseFaissRetriever(RetrieverPort):
    """Busca por similitud densa usando un índice FAISS.

    Parameters
    ----------
    embedder:
        Implementación de :class:`EmbedderPort` para transformar texto→vector.
        Si es *None*, se usa :class:`SentenceTransformerEmbedder` por defecto.
    index_path / id_map_path:
        Ubicaciones de los artefactos. Por defecto se leen de ``settings``.
    """

    def __init__(
        self,
        *,
        embedder: EmbedderPort | None = None,
        index_path: str | Path | None = None,
        id_map_path: str | Path | None = None,
    ) -> None:
        # --- Embedder -----------------------------------------------------
        self.embedder: EmbedderPort = embedder or SentenceTransformerEmbedder()

        # --- Rutas de artefactos -----------------------------------------
        self.index_path = Path(index_path or settings.index_path)
        self.id_map_path = Path(id_map_path or settings.id_map_path)

        if not self.index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.id_map_path.is_file():
            raise FileNotFoundError(f"ID‑map file not found: {self.id_map_path}")

        # --- Cargar índice FAISS -----------------------------------------
        self.index: faiss.Index = faiss.read_index(str(self.index_path))

        # --- Cargar mapa de IDs ------------------------------------------
        with self.id_map_path.open("rb") as fh:
            raw_id_map = pickle.load(fh)

        # Permitimos lista o dict para compatibilidad hacia atrás
        if isinstance(raw_id_map, dict):
            max_idx = max(raw_id_map.keys(), default=-1)
            id_map: list[int | None] = [None] * (max_idx + 1)
            for k, v in raw_id_map.items():
                id_map[k] = v
            self._id_map: list[int | None] = id_map
        elif isinstance(raw_id_map, Sequence):
            self._id_map = list(raw_id_map)  # shallow copy por seguridad
        else:
            raise TypeError(
                "id_map must be a list or dict mapping faiss_idx→doc_id"
            )

        # --- Validaciones -------------------------------------------------
        if self.index.d != self.embedder.DIM:
            raise ValueError(
                "Dimension mismatch: index «%d» vs embedder «%d»"
                % (self.index.d, self.embedder.DIM)
            )

    # ---------------------------------------------------------------------
    # API pública
    # ---------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        """Devuelve los *k* documentos más similares.

        Retorna dos listas paralelas (ids, scores). Si ``k<=0`` se devuelven
        listas vacías.
        """
        if k <= 0:
            return [], []

        # FAISS espera shape (1, dim)
        q_vec = np.asarray([self.embedder.embed(query)], dtype="float32")
        sims, idxs = self.index.search(q_vec, min(k, self.index.ntotal))

        sims = sims[0]
        idxs = idxs[0]

        ids: list[int] = []
        scores: list[float] = []
        for faiss_idx, score in zip(idxs, sims):
            if faiss_idx == -1:
                continue  # FAISS rellena con -1 si no alcanza k resultados
            doc_id = self._id_map[faiss_idx]
            if doc_id is None:
                continue  # hueco en el mapa
            ids.append(int(doc_id))
            scores.append(float(score))

        return ids, scores
