"""Content-addressed persistent embedding cache for one active embedding model."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from local_rag_backend.core.ports import EmbedderPort
from local_rag_backend.infrastructure.observability.perf import (
    record_embedding_cache_embed,
    record_embedding_cache_error,
    record_embedding_cache_lookup,
    record_embedding_cache_store,
)

if TYPE_CHECKING:
    from local_rag_backend.core.domain.entities import Embedding


def resolve_embedding_model_key(embedder: EmbedderPort) -> str:
    backend = "openai" if hasattr(embedder, "client") else "sentence_transformers"
    model_name = getattr(embedder, "model_name", None) or getattr(embedder, "model", None)
    model = str(model_name or embedder.__class__.__name__)
    return f"{backend}:{model}:{int(embedder.dim)}"


def resolve_embedding_cache_db_path(
    *, data_dir: Path, configured_path: str | Path | None = None
) -> Path:
    if configured_path is not None and str(configured_path).strip():
        return Path(configured_path)
    return data_dir / "embedding_cache.sqlite3"


class _SqliteEmbeddingCache:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                model_key TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY (model_key, content_sha256)
            )
            """
        )
        return conn

    def get_many(self, *, model_key: str, content_hashes: Sequence[str]) -> dict[str, list[float]]:
        hashes = [str(item) for item in content_hashes if str(item).strip()]
        if not hashes:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT content_sha256, vector_json
                FROM embedding_cache
                WHERE model_key = ?
                  AND content_sha256 IN (
                      SELECT value
                      FROM json_each(?)
                  )
                """,
                [model_key, json.dumps(hashes)],
            ).fetchall()
        return {str(content_sha): list(json.loads(vector_json)) for content_sha, vector_json in rows}

    def put_many(self, *, model_key: str, vectors_by_hash: Mapping[str, Sequence[float]]) -> None:
        rows = [
            (model_key, str(content_sha), json.dumps(list(vector)))
            for content_sha, vector in vectors_by_hash.items()
        ]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (model_key, content_sha256, vector_json)
                VALUES (?, ?, ?)
                """,
                rows,
            )
            conn.commit()


class ContentAddressedCachingEmbedder(EmbedderPort):
    """Best-effort persistent cache around a dense embedder."""

    dim: int

    def __init__(
        self,
        *,
        base: EmbedderPort,
        cache_db_path: Path,
        disabled: bool = False,
    ) -> None:
        self._base = base
        self._cache = _SqliteEmbeddingCache(cache_db_path)
        self._disabled = bool(disabled)
        self.dim = base.dim
        self.model_key = resolve_embedding_model_key(base)

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]:
        texts_list = [str(text) for text in texts]
        if not texts_list or self._disabled:
            return self._base.embed(texts_list)
        try:
            hashes_in_order: list[str] = []
            by_hash: dict[str, str] = {}
            for text in texts_list:
                content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                hashes_in_order.append(content_sha)
                by_hash.setdefault(content_sha, text)

            lookup_start = time.perf_counter()
            cached = self._cache.get_many(
                model_key=self.model_key,
                content_hashes=tuple(by_hash),
            )
            lookup_seconds = time.perf_counter() - lookup_start
            misses = [content_sha for content_sha in by_hash if content_sha not in cached]
            record_embedding_cache_lookup(
                hits=len(by_hash) - len(misses),
                misses=len(misses),
                seconds=lookup_seconds,
            )

            if misses:
                miss_texts = [by_hash[content_sha] for content_sha in misses]
                embed_start = time.perf_counter()
                embedded = self._base.embed(miss_texts)
                embed_seconds = time.perf_counter() - embed_start
                record_embedding_cache_embed(count=len(miss_texts), seconds=embed_seconds)
                if len(embedded) != len(miss_texts):
                    raise RuntimeError(
                        f"Embedder returned {len(embedded)} vectors for {len(miss_texts)} texts."
                    )
                miss_vectors = {
                    content_sha: list(vector)
                    for content_sha, vector in zip(misses, embedded, strict=False)
                }
                store_start = time.perf_counter()
                self._cache.put_many(model_key=self.model_key, vectors_by_hash=miss_vectors)
                record_embedding_cache_store(
                    count=len(miss_vectors),
                    seconds=time.perf_counter() - store_start,
                )
                cached.update(miss_vectors)

            return cast("Sequence[Embedding]", [list(cached[content_sha]) for content_sha in hashes_in_order])
        except Exception:
            record_embedding_cache_error()
            return self._base.embed(texts_list)
