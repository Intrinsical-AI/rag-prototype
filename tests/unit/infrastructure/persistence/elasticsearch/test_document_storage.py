from __future__ import annotations

import json

import httpx

from local_rag_backend.infrastructure.persistence.elasticsearch.client import ElasticClient
from local_rag_backend.infrastructure.persistence.elasticsearch.document_storage import (
    ElasticDocsRepository,
)
from local_rag_backend.settings import Settings


def _build_transport() -> httpx.MockTransport:
    indices: dict[str, dict[str, object]] = {}

    def _json(request: httpx.Request) -> dict[str, object]:
        content = request.content.decode("utf-8")
        return dict(json.loads(content)) if content else {}

    def _bulk_ops(request: httpx.Request) -> list[dict[str, object]]:
        lines = [line for line in request.content.decode("utf-8").splitlines() if line.strip()]
        return [dict(json.loads(line)) for line in lines]

    def _docs(index: str) -> dict[str, dict[str, object]]:
        bucket = indices.setdefault(index, {"docs": {}, "mapping": {}})
        return bucket["docs"]  # type: ignore[return-value]

    def _mapping(index: str) -> dict[str, object]:
        bucket = indices.setdefault(index, {"docs": {}, "mapping": {}})
        return bucket["mapping"]  # type: ignore[return-value]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if method == "HEAD":
            index = path.strip("/")
            return httpx.Response(200 if index in indices else 404)

        if method == "GET" and path == "/":
            return httpx.Response(200, json={"status": "ok"})

        if method == "PUT" and path.count("/") == 1:
            index = path.strip("/")
            body = _json(request)
            mapping = (((body.get("mappings") or {}).get("properties")) or {})
            indices[index] = {"docs": {}, "mapping": dict(mapping)}
            return httpx.Response(200, json={"acknowledged": True})

        if method == "PUT" and path.endswith("/_mapping"):
            index = path.strip("/").split("/")[0]
            body = _json(request)
            _mapping(index).update(dict(body.get("properties") or {}))
            return httpx.Response(200, json={"acknowledged": True})

        if method == "GET" and path.endswith("/_mapping"):
            index = path.strip("/").split("/")[0]
            return httpx.Response(
                200,
                json={index: {"mappings": {"properties": dict(_mapping(index))}}},
            )

        if method == "POST" and path.endswith("/_mget"):
            index = path.strip("/").split("/")[0]
            docs = _docs(index)
            body = _json(request)
            found = []
            for doc_id in list(body.get("ids") or []):
                source = docs.get(str(doc_id))
                if source is None:
                    found.append({"_id": str(doc_id), "found": False})
                else:
                    found.append({"_id": str(doc_id), "found": True, "_source": dict(source)})
            return httpx.Response(200, json={"docs": found})

        if method == "POST" and path == "/_bulk":
            ops = _bulk_ops(request)
            i = 0
            while i < len(ops):
                op = ops[i]
                if "index" in op:
                    meta = dict(op["index"])
                    body = dict(ops[i + 1])
                    _docs(str(meta["_index"]))[str(meta["_id"])] = body
                    i += 2
                    continue
                if "delete" in op:
                    meta = dict(op["delete"])
                    _docs(str(meta["_index"])).pop(str(meta["_id"]), None)
                    i += 1
                    continue
                raise AssertionError(f"unexpected bulk op: {op}")
            return httpx.Response(200, json={"errors": False})

        if method == "POST" and path.endswith("/_search"):
            index = path.strip("/").split("/")[0]
            docs = _docs(index)
            body = _json(request)
            query = dict(body.get("query") or {})
            sort = list(body.get("sort") or [])
            term_scope = ((query.get("term") or {}).get("scope")) if isinstance(query, dict) else None
            prefix_external_id = (
                ((query.get("prefix") or {}).get("external_id")) if isinstance(query, dict) else None
            )
            hits = []
            for doc_id, source in docs.items():
                if term_scope is not None and source.get("scope") != term_scope:
                    continue
                if prefix_external_id is not None and not str(source.get("external_id") or "").startswith(
                    str(prefix_external_id)
                ):
                    continue
                hits.append({"_id": str(doc_id), "_source": dict(source), "sort": [source.get("external_id")]})
            if sort:
                hits.sort(key=lambda hit: str((hit.get("_source") or {}).get("external_id") or hit.get("_id") or ""))
            return httpx.Response(200, json={"hits": {"hits": hits}})

        raise AssertionError(f"Unhandled {method} {path}")

    return httpx.MockTransport(handler)


def _build_repo() -> ElasticDocsRepository:
    settings_obj = Settings(
        persistence_backend="elasticsearch",
        retrieval_mode="dense",
        es_base_url="http://example.test",
    )
    client = ElasticClient(
        settings_obj=settings_obj,
        client=httpx.Client(base_url="http://example.test", transport=_build_transport()),
    )
    return ElasticDocsRepository(settings_obj=settings_obj, client=client)


def test_upsert_is_unchanged_when_metadata_matches() -> None:
    repo = _build_repo()
    item = ElasticDocsRepository.UpsertDoc(
        external_id="doc-1",
        content="hello world",
        source_id="src-1",
        metadata={"kind": "faq"},
        chunk_dedup_sha256="dedup-1",
    )

    first_results, _first_changed, _first_updated = repo.upsert_documents_by_external_id([item])
    second_results, second_changed, second_updated = repo.upsert_documents_by_external_id([item])

    assert first_results[0].action == "inserted"
    assert second_results[0].action == "unchanged"
    assert second_changed == []
    assert second_updated == []


def test_delete_by_external_id_creates_tombstone() -> None:
    repo = _build_repo()
    repo.upsert_documents_by_external_id(
        [
            ElasticDocsRepository.UpsertDoc(
                external_id="doc-1",
                content="hello world",
            )
        ]
    )

    deleted_count, deleted_ids, missing, tombstoned = repo.delete_by_external_ids(["doc-1"])

    assert deleted_count == 1
    assert [str(doc_id) for doc_id in deleted_ids] == ["doc-1"]
    assert missing == []
    assert tombstoned == 1
    assert repo.get_tombstoned_external_ids(["doc-1"]) == {"doc-1"}


def test_scope_and_snapshot_are_persisted_and_queryable() -> None:
    repo = _build_repo()
    repo.upsert_documents_by_external_id(
        [
            ElasticDocsRepository.UpsertDoc(
                external_id="doc-1",
                content="hello world",
                scope="repogpt:demo",
                snapshot_id="snap-1",
                metadata={"kind": "code"},
            )
        ]
    )

    doc = repo.get(["doc-1"])[0]
    assert doc.metadata == {
        "kind": "code",
        "scope": "repogpt:demo",
        "snapshot_id": "snap-1",
    }
    assert repo.list_external_ids_by_scope("repogpt:demo") == ["doc-1"]
