"""History storage backed by Elasticsearch."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.ports import QAHistoryPort
from local_rag_backend.infrastructure.persistence.elasticsearch.client import ElasticClient
from local_rag_backend.settings import Settings, settings as global_settings


@dataclass(frozen=True)
class ElasticHistoryEntry:
    id: int
    question: str
    answer: str
    created_at: str
    source_ids: tuple[str, ...]


class ElasticHistoryStorage(QAHistoryPort):
    def __init__(
        self,
        *,
        settings_obj: Settings | None = None,
        client: ElasticClient | None = None,
    ) -> None:
        self._settings = settings_obj or global_settings
        self._client = client or ElasticClient(settings_obj=self._settings)
        self._client.ensure_indices(embed_dim=None)

    def save(self, q: str, a: str, source_ids: Sequence[DocId]) -> None:
        created_at = datetime.now(UTC).isoformat()
        history_id = int(time.time_ns())
        self._client.index_document(
            index=str(self._settings.es_history_index),
            id_=str(history_id),
            body={
                "id": history_id,
                "question": str(q),
                "answer": str(a),
                "created_at": created_at,
                "source_ids": [str(x) for x in source_ids],
            },
        )

    def list_entries(self, *, limit: int, offset: int) -> tuple[ElasticHistoryEntry, ...]:
        body = {
            "size": int(limit),
            "from": int(offset),
            "sort": [{"created_at": {"order": "desc"}}],
            "query": {"match_all": {}},
        }
        data = self._client.search(index=str(self._settings.es_history_index), body=body)
        hits = (((data.get("hits") or {}).get("hits")) or [])
        return tuple(
            ElasticHistoryEntry(
                id=int((hit.get("_source") or {}).get("id") or 0),
                question=str((hit.get("_source") or {}).get("question") or ""),
                answer=str((hit.get("_source") or {}).get("answer") or ""),
                created_at=str((hit.get("_source") or {}).get("created_at") or ""),
                source_ids=tuple(str(x) for x in ((hit.get("_source") or {}).get("source_ids") or [])),
            )
            for hit in hits
            if isinstance(hit, dict)
        )


__all__ = ["ElasticHistoryEntry", "ElasticHistoryStorage"]
