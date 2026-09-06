from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from local_rag_backend.core.use_cases.docs_import_canonical import _delete_stale_scope_documents
from local_rag_backend.infrastructure.persistence.elasticsearch.client import (
    ElasticBackendError,
    ElasticClient,
)
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
            mapping = ((body.get("mappings") or {}).get("properties")) or {}
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
            items = []
            i = 0
            while i < len(ops):
                op = ops[i]
                if "index" in op:
                    meta = dict(op["index"])
                    body = dict(ops[i + 1])
                    _docs(str(meta["_index"]))[str(meta["_id"])] = body
                    items.append({"index": {"_id": str(meta["_id"]), "status": 201}})
                    i += 2
                    continue
                if "delete" in op:
                    meta = dict(op["delete"])
                    _docs(str(meta["_index"])).pop(str(meta["_id"]), None)
                    items.append({"delete": {"_id": str(meta["_id"]), "status": 200}})
                    i += 1
                    continue
                raise AssertionError(f"unexpected bulk op: {op}")
            return httpx.Response(200, json={"errors": False, "items": items})

        if method == "POST" and path.endswith("/_search"):
            index = path.strip("/").split("/")[0]
            docs = _docs(index)
            body = _json(request)
            query = dict(body.get("query") or {})
            sort = list(body.get("sort") or [])
            term_scope = (
                ((query.get("term") or {}).get("scope")) if isinstance(query, dict) else None
            )
            prefix_external_id = (
                ((query.get("prefix") or {}).get("external_id"))
                if isinstance(query, dict)
                else None
            )
            hits = []
            for doc_id, source in docs.items():
                if term_scope is not None and source.get("scope") != term_scope:
                    continue
                if prefix_external_id is not None and not str(
                    source.get("external_id") or ""
                ).startswith(str(prefix_external_id)):
                    continue
                hits.append(
                    {
                        "_id": str(doc_id),
                        "_source": dict(source),
                        "sort": [source.get("external_id")],
                    }
                )
            if sort:
                hits.sort(
                    key=lambda hit: str(
                        (hit.get("_source") or {}).get("external_id") or hit.get("_id") or ""
                    )
                )
            if body.get("search_after"):
                hits = [hit for hit in hits if hit["sort"] > body["search_after"]]
            hits = hits[: int(body.get("size", 10))]
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


def test_canonical_scope_delete_removes_colocated_embedding_without_vector_update():
    repo = _build_repo()
    repo.upsert_documents_by_external_id(
        [
            repo.UpsertDoc(external_id="stale", content="old", scope="demo", embedding=[1.0, 0.0]),
            repo.UpsertDoc(external_id="keep", content="new", scope="demo", embedding=[0.0, 1.0]),
        ]
    )

    def vector_factory(**kwargs):
        raise AssertionError("Embedding is deleted with its Elasticsearch document")

    kwargs = {
        "scope": "demo",
        "keep_external_ids": {"keep"},
        "settings_obj": repo._settings,
        "ports": SimpleNamespace(doc_repo_factory=lambda: repo, vector_repo_factory=vector_factory),
    }
    assert _delete_stale_scope_documents(**kwargs) == (["stale"], 1, 1)
    assert [doc.external_id for doc in repo.get(["keep", "stale"])] == ["keep"]
    assert not repo.get_tombstoned_external_ids(["stale"])
    assert _delete_stale_scope_documents(**kwargs) == ([], 0, 0)


@pytest.mark.parametrize("count", [1000, 1002, 2503])
def test_scope_cleanup_enumerates_all_pages_before_deleting(count, monkeypatch):
    repo = _build_repo()
    ids = [f"doc-{i:05d}" for i in range(count)]
    repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id=id_, content=id_, scope="demo") for id_ in ids]
        + [repo.UpsertDoc(external_id="foreign", content="keep other scope", scope="other")]
    )
    events = []
    search, bulk = repo._client.search, repo._client.bulk

    def observe_search(**kwargs):
        events.append(("search", kwargs["body"].get("search_after")))
        return search(**kwargs)

    def observe_bulk(*args, **kwargs):
        events.append(("bulk", None))
        return bulk(*args, **kwargs)

    monkeypatch.setattr(repo._client, "search", observe_search)
    monkeypatch.setattr(repo._client, "bulk", observe_bulk)
    kwargs = {
        "scope": "demo",
        "keep_external_ids": set(ids[:1000]),
        "settings_obj": repo._settings,
        "ports": SimpleNamespace(doc_repo_factory=lambda: repo),
    }
    stale, deleted, vectors = _delete_stale_scope_documents(**kwargs)
    assert stale == ids[1000:]
    assert deleted == vectors == max(0, count - 1000)
    expected_pages = count // 1000 + 1
    assert events[:expected_pages] == [
        ("search", None if page == 0 else [ids[page * 1000 - 1]]) for page in range(expected_pages)
    ]
    assert all(event[0] == "bulk" for event in events[expected_pages:])
    assert repo.list_external_ids_by_scope("demo") == ids[:1000]
    assert repo.get(["foreign"])[0].content == "keep other scope"
    assert _delete_stale_scope_documents(**kwargs) == ([], 0, 0)
    assert not repo.get_tombstoned_external_ids(stale)


@pytest.mark.parametrize("bad_token", [None, [], [None], [1], "cursor", ["a", "b"], ["doc-00999"]])
def test_incomplete_or_repeated_scope_cursor_fails_before_deletion(bad_token, monkeypatch):
    repo = _build_repo()
    repo.upsert_documents_by_external_id(
        [
            repo.UpsertDoc(external_id=f"doc-{i:05d}", content="old", scope="demo")
            for i in range(2001)
        ]
    )
    search = repo._client.search

    def broken_page(**kwargs):
        result = search(**kwargs)
        # The second full page cannot be mistaken for successful enumeration.
        if kwargs["body"].get("search_after"):
            result["hits"]["hits"][-1]["sort"] = bad_token
        return result

    def no_deletion(*args, **kwargs):
        pytest.fail("Deletion started before scope enumeration completed")

    monkeypatch.setattr(repo._client, "search", broken_page)
    monkeypatch.setattr(repo._client, "bulk", no_deletion)
    with pytest.raises(ElasticBackendError, match="pagination"):
        _delete_stale_scope_documents(
            scope="demo",
            keep_external_ids={"doc-00000"},
            settings_obj=repo._settings,
            ports=SimpleNamespace(doc_repo_factory=lambda: repo),
        )


def _partial_markers(mode: str) -> dict[str, object]:
    """Response fields Elasticsearch uses to declare a search incomplete."""
    if mode == "timed_out":
        return {"timed_out": True}
    return {"_shards": {"total": 2, "successful": 1, "skipped": 0, "failed": 1}}


PARTIAL_MODES = ["timed_out", "shard_failure"]


@pytest.mark.parametrize("mode", PARTIAL_MODES)
def test_partial_scope_search_fails_before_any_deletion(mode, monkeypatch):
    repo = _build_repo()
    repo.upsert_documents_by_external_id(
        [
            repo.UpsertDoc(external_id="keep", content="keep", scope="demo"),
            repo.UpsertDoc(external_id="stale", content="stale", scope="demo"),
        ]
    )
    search = repo._client.search

    def partial_page(**kwargs):
        result = search(**kwargs)
        # Elasticsearch silently drops the hidden document from a partial response.
        result["hits"]["hits"] = [hit for hit in result["hits"]["hits"] if hit["_id"] != "stale"]
        result.update(_partial_markers(mode))
        return result

    def no_deletion(*args, **kwargs):
        pytest.fail("Deletion started from a partial scope enumeration")

    monkeypatch.setattr(repo._client, "search", partial_page)
    monkeypatch.setattr(repo._client, "bulk", no_deletion)
    with pytest.raises(ElasticBackendError, match="partial"):
        _delete_stale_scope_documents(
            scope="demo",
            keep_external_ids={"keep"},
            settings_obj=repo._settings,
            ports=SimpleNamespace(doc_repo_factory=lambda: repo),
        )


@pytest.mark.parametrize("mode", PARTIAL_MODES)
def test_scope_search_turning_partial_after_a_full_page_fails_before_deletion(mode, monkeypatch):
    repo = _build_repo()
    ids = [f"doc-{i:05d}" for i in range(1500)]
    repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id=id_, content=id_, scope="demo") for id_ in ids]
    )
    search = repo._client.search

    def partial_second_page(**kwargs):
        result = search(**kwargs)
        # The first page is complete, so truncation can only be seen on the second.
        if kwargs["body"].get("search_after"):
            result.update(_partial_markers(mode))
        return result

    def no_deletion(*args, **kwargs):
        pytest.fail("Deletion started from a partial scope enumeration")

    monkeypatch.setattr(repo._client, "search", partial_second_page)
    monkeypatch.setattr(repo._client, "bulk", no_deletion)
    with pytest.raises(ElasticBackendError, match="partial"):
        _delete_stale_scope_documents(
            scope="demo",
            keep_external_ids=set(ids[:1000]),
            settings_obj=repo._settings,
            ports=SimpleNamespace(doc_repo_factory=lambda: repo),
        )


def test_scope_search_requests_strict_completeness():
    repo = _build_repo()
    seen = []
    search = repo._client.search

    def record_params(**kwargs):
        seen.append(kwargs.get("params"))
        return search(**kwargs)

    repo._client.search = record_params
    repo.list_external_ids_by_scope("demo")
    repo.get_all_documents()
    assert seen == [{"allow_partial_search_results": "false"}] * 2


@pytest.mark.parametrize("mode", PARTIAL_MODES)
def test_partial_document_scan_is_rejected(mode, monkeypatch):
    repo = _build_repo()
    repo.upsert_documents_by_external_id([repo.UpsertDoc(external_id="doc-1", content="hello")])
    search = repo._client.search

    def partial_page(**kwargs):
        result = search(**kwargs)
        result.update(_partial_markers(mode))
        return result

    monkeypatch.setattr(repo._client, "search", partial_page)
    with pytest.raises(ElasticBackendError, match="partial"):
        repo.get_all_documents()


def test_document_scan_rejects_full_page_without_cursor(monkeypatch):
    repo = _build_repo()
    ids = [f"doc-{i:05d}" for i in range(600)]
    repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id=id_, content=id_) for id_ in ids]
    )
    search = repo._client.search

    def cursorless_page(**kwargs):
        result = search(**kwargs)
        result["hits"]["hits"][-1].pop("sort", None)
        return result

    monkeypatch.setattr(repo._client, "search", cursorless_page)
    with pytest.raises(ElasticBackendError, match="cursor"):
        repo.get_all_documents()


def test_document_scan_rejects_malformed_hits(monkeypatch):
    repo = _build_repo()
    repo.upsert_documents_by_external_id([repo.UpsertDoc(external_id="doc-1", content="hello")])
    monkeypatch.setattr(repo._client, "search", lambda **kwargs: {"hits": {"hits": "not-a-list"}})
    with pytest.raises(ElasticBackendError, match="document scan"):
        repo.get_all_documents()


def test_complete_multi_page_scan_returns_every_document():
    repo = _build_repo()
    ids = [f"doc-{i:05d}" for i in range(1200)]
    repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id=id_, content=id_) for id_ in ids]
    )
    assert sorted(doc.external_id for doc in repo.get_all_documents()) == ids


@pytest.mark.parametrize("errors", [True, False])
def test_bulk_reports_individual_failures_even_when_http_succeeds(errors):
    response = {
        "errors": errors,
        "items": [
            {"delete": {"_id": "good", "status": 200}},
            {
                "delete": {
                    "_id": "bad",
                    "status": 429,
                    "error": {"type": "rejected_execution_exception"},
                }
            },
        ],
    }
    client = ElasticClient(
        client=httpx.Client(
            base_url="http://example.test",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
        )
    )
    with pytest.raises(ElasticBackendError, match="delete bad: status 429"):
        client.bulk(
            [
                {"delete": {"_index": "docs", "_id": "good"}},
                {"delete": {"_index": "docs", "_id": "bad"}},
            ]
        )


def test_bulk_delete_absent_document_is_idempotent():
    client = ElasticClient(
        client=httpx.Client(
            base_url="http://example.test",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "errors": False,
                        "items": [
                            {"delete": {"_id": "missing", "status": 404, "result": "not_found"}}
                        ],
                    },
                )
            ),
        )
    )
    assert not client.bulk([{"delete": {"_index": "docs", "_id": "missing"}}])["errors"]


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
