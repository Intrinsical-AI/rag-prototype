"""System-state storage backed by Elasticsearch."""

from __future__ import annotations

from datetime import UTC, datetime

from local_rag_backend.infrastructure.persistence.elasticsearch.client import ElasticClient
from local_rag_backend.settings import Settings, settings as global_settings


class ElasticSystemStateStorage:
    def __init__(
        self,
        *,
        settings_obj: Settings | None = None,
        client: ElasticClient | None = None,
    ) -> None:
        self._settings = settings_obj or global_settings
        self._client = client or ElasticClient(settings_obj=self._settings)
        self._client.ensure_indices(embed_dim=None)

    def get_version(self, key: str) -> int:
        state_key = str(key).strip()
        if not state_key:
            raise ValueError("system_state key must not be blank")
        docs = self._client.mget(index=str(self._settings.es_system_index), ids=[state_key])
        if not docs or not bool(docs[0].get("found")):
            return 0
        source = dict(docs[0].get("_source") or {})
        return int(source.get("version") or 0)

    def bump_version(self, key: str) -> int:
        state_key = str(key).strip()
        if not state_key:
            raise ValueError("system_state key must not be blank")
        current = self.get_version(state_key)
        new_version = current + 1
        self._client.index_document(
            index=str(self._settings.es_system_index),
            id_=state_key,
            body={
                "key": state_key,
                "version": new_version,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        return new_version


__all__ = ["ElasticSystemStateStorage"]
