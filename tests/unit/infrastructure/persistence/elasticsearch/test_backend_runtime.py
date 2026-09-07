from __future__ import annotations

import json
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from local_rag_backend.core.domain.profiles import StorageProfileRegistry
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.ports.contracts import DocsMutationPorts
from local_rag_backend.core.use_cases._atomic_mutation_executor import AtomicMutationExecutor
from local_rag_backend.core.use_cases._mutation_saga_executor import PreparedMutation
from local_rag_backend.core.use_cases.docs_import_canonical import (
    CanonicalImportDocumentInput,
    CanonicalImportRequestInput,
    execute_import_canonical_sync,
)
from local_rag_backend.core.use_cases.docs_mutation_contracts import (
    MutationIntent,
    MutationUpsertInput,
)
from local_rag_backend.infrastructure.persistence.elasticsearch.client import (
    ElasticBackendError,
    ElasticClient,
)
from local_rag_backend.infrastructure.persistence.elasticsearch.diagnostics import (
    ElasticHealthDiagnostics,
    purge_index_artifacts_noop,
)
from local_rag_backend.infrastructure.persistence.elasticsearch.document_storage import (
    ElasticDocsRepository,
    ElasticVectorRepo,
)
from local_rag_backend.infrastructure.persistence.elasticsearch.history_storage import (
    ElasticHistoryStorage,
)
from local_rag_backend.infrastructure.persistence.elasticsearch.system_state import (
    ElasticSystemStateStorage,
)
from local_rag_backend.settings import Settings


def _score_text(query: str, text: str) -> float:
    tokens = [token for token in query.lower().split() if token]
    haystack = text.lower()
    return float(sum(1 for token in tokens if token in haystack))


@dataclass
class _ElasticTestState:
    indices: dict[str, dict[str, Any]]

    def __init__(self) -> None:
        self.indices = {}

    def docs_for(self, index: str) -> dict[str, dict[str, Any]]:
        bucket = self.indices.setdefault(index, {"docs": {}, "mapping": {}})
        return bucket["docs"]  # type: ignore[return-value]

    def mapping_for(self, index: str) -> dict[str, Any]:
        bucket = self.indices.setdefault(index, {"docs": {}, "mapping": {}})
        return bucket["mapping"]  # type: ignore[return-value]

    def transport(self) -> httpx.MockTransport:
        def _json(request: httpx.Request) -> dict[str, Any]:
            content = request.content.decode("utf-8")
            return dict(json.loads(content)) if content else {}

        def _bulk_ops(request: httpx.Request) -> list[dict[str, Any]]:
            lines = [line for line in request.content.decode("utf-8").splitlines() if line.strip()]
            return [dict(json.loads(line)) for line in lines]

        def _cosine(query_vector: list[float], vector: list[float]) -> float:
            dot = sum(a * b for a, b in zip(query_vector, vector, strict=False))
            lhs = math.sqrt(sum(a * a for a in query_vector))
            rhs = math.sqrt(sum(b * b for b in vector))
            if lhs == 0.0 or rhs == 0.0:
                return 0.0
            return dot / (lhs * rhs)

        def _search_hits(index: str, body: dict[str, Any]) -> list[dict[str, Any]]:
            docs = self.docs_for(index)
            sort_spec = list(body.get("sort") or [])
            include_source = body.get("_source", True) is not False
            hits: list[dict[str, Any]] = []

            knn = body.get("knn")
            if isinstance(knn, dict):
                field = str(knn.get("field") or "")
                query_vector = [float(x) for x in list(knn.get("query_vector") or [])]
                size = int(body.get("size") or knn.get("k") or 10)
                for doc_id, source in docs.items():
                    vector = source.get(field)
                    if not isinstance(vector, list):
                        continue
                    score = _cosine(query_vector, [float(x) for x in vector])
                    hits.append({"_id": doc_id, "_source": dict(source), "_score": score})
                hits.sort(key=lambda hit: float(hit.get("_score") or 0.0), reverse=True)
                hits = hits[:size]
            else:
                query = dict(body.get("query") or {"match_all": {}})
                for doc_id, source in docs.items():
                    score = 1.0
                    if "match_all" in query:
                        pass
                    elif "term" in query:
                        field, value = next(iter(query["term"].items()))
                        if source.get(field) != value:
                            continue
                    elif "prefix" in query:
                        field, prefix = next(iter(dict(query["prefix"]).items()))
                        if not str(source.get(str(field)) or doc_id).startswith(str(prefix)):
                            continue
                    elif "exists" in query:
                        field = str((query["exists"] or {}).get("field") or "")
                        if field not in source or source.get(field) is None:
                            continue
                    elif "match" in query:
                        field, raw_value = next(iter(dict(query["match"]).items()))
                        search_query = (
                            str((raw_value or {}).get("query") or "")
                            if isinstance(raw_value, dict)
                            else str(raw_value)
                        )
                        score = _score_text(search_query, str(source.get(str(field)) or ""))
                        if score <= 0.0:
                            continue
                    else:
                        raise AssertionError(f"Unsupported query: {query}")
                    hits.append({"_id": doc_id, "_source": dict(source), "_score": score})

                if sort_spec:
                    sort_entry = dict(sort_spec[0])
                    field, order = next(iter(sort_entry.items()))
                    reverse = False
                    if isinstance(order, dict):
                        reverse = str(order.get("order") or "asc").lower() == "desc"
                    else:
                        reverse = str(order).lower() == "desc"
                    hits.sort(
                        key=lambda hit: str(
                            (hit.get("_source") or {}).get(str(field)) or hit.get("_id") or ""
                        ),
                        reverse=reverse,
                    )
                    search_after = list(body.get("search_after") or [])
                    if search_after:
                        cursor = str(search_after[0])
                        hits = [
                            hit
                            for hit in hits
                            if str(
                                (hit.get("_source") or {}).get(str(field)) or hit.get("_id") or ""
                            )
                            > cursor
                        ]
                elif hits and any(float(hit.get("_score") or 0.0) != 1.0 for hit in hits):
                    hits.sort(key=lambda hit: float(hit.get("_score") or 0.0), reverse=True)

                if "from" in body:
                    hits = hits[int(body.get("from") or 0) :]
                if "size" in body:
                    hits = hits[: int(body.get("size") or 10)]

            out: list[dict[str, Any]] = []
            for hit in hits:
                entry: dict[str, Any] = {"_id": str(hit["_id"]), "_score": float(hit["_score"])}
                if include_source:
                    entry["_source"] = dict(hit["_source"])
                if sort_spec:
                    field = next(iter(dict(sort_spec[0]).keys()))
                    entry["sort"] = [
                        str((hit.get("_source") or {}).get(str(field)) or hit.get("_id") or "")
                    ]
                out.append(entry)
            return out

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            method = request.method

            if method == "HEAD":
                index = path.strip("/")
                return httpx.Response(200 if index in self.indices else 404)

            if method == "GET" and path == "/":
                return httpx.Response(200, json={"status": "ok"})

            if method == "PUT" and path.count("/") == 1:
                index = path.strip("/")
                body = _json(request)
                mapping = dict(((body.get("mappings") or {}).get("properties")) or {})
                self.indices[index] = {"docs": {}, "mapping": mapping}
                return httpx.Response(200, json={"acknowledged": True})

            if method == "PUT" and path.endswith("/_mapping"):
                index = path.strip("/").split("/")[0]
                body = _json(request)
                self.mapping_for(index).update(dict(body.get("properties") or {}))
                return httpx.Response(200, json={"acknowledged": True})

            if method == "GET" and path.endswith("/_mapping"):
                index = path.strip("/").split("/")[0]
                return httpx.Response(
                    200,
                    json={index: {"mappings": {"properties": dict(self.mapping_for(index))}}},
                )

            if method == "POST" and path.endswith("/_mget"):
                index = path.strip("/").split("/")[0]
                docs = self.docs_for(index)
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
                items: list[dict[str, Any]] = []
                i = 0
                while i < len(ops):
                    op = ops[i]
                    if "index" in op:
                        meta = dict(op["index"])
                        body = dict(ops[i + 1])
                        self.docs_for(str(meta["_index"]))[str(meta["_id"])] = body
                        items.append({"index": {"_id": str(meta["_id"]), "status": 201}})
                        i += 2
                        continue
                    if "delete" in op:
                        meta = dict(op["delete"])
                        self.docs_for(str(meta["_index"])).pop(str(meta["_id"]), None)
                        items.append({"delete": {"_id": str(meta["_id"]), "status": 200}})
                        i += 1
                        continue
                    if "update" in op:
                        meta = dict(op["update"])
                        body = dict(ops[i + 1])
                        index = str(meta["_index"])
                        doc = (
                            self.docs_for(index).get(str(meta["_id"]))
                            if index in self.indices
                            else None
                        )
                        if doc is None:
                            error_type = (
                                "document_missing_exception"
                                if index in self.indices
                                else "index_not_found_exception"
                            )
                            items.append(
                                {
                                    "update": {
                                        "_id": str(meta["_id"]),
                                        "status": 404,
                                        "error": {"type": error_type},
                                    }
                                }
                            )
                            i += 2
                            continue
                        if "script" in body:
                            source = str((body["script"] or {}).get("source") or "")
                            match = re.search(r"remove\('([^']+)'\)", source)
                            if match:
                                doc.pop(match.group(1), None)
                        elif "doc" in body:
                            doc.update(dict(body.get("doc") or {}))
                        else:
                            raise AssertionError(f"Unsupported update body: {body}")
                        items.append({"update": {"_id": str(meta["_id"]), "status": 200}})
                        i += 2
                        continue
                    raise AssertionError(f"Unexpected bulk op: {op}")
                errors = any(outcome.get("error") for item in items for outcome in item.values())
                return httpx.Response(200, json={"errors": errors, "items": items})

            if method == "POST" and path.endswith("/_search"):
                index = path.strip("/").split("/")[0]
                body = _json(request)
                hits = _search_hits(index, body)
                return httpx.Response(200, json={"hits": {"hits": hits}})

            if method == "POST" and path.endswith("/_count"):
                index = path.strip("/").split("/")[0]
                body = _json(request)
                hits = _search_hits(index, {"query": body.get("query") or {"match_all": {}}})
                return httpx.Response(200, json={"count": len(hits)})

            if method == "DELETE" and "/_doc/" in path:
                index, doc_id = path.strip("/").split("/_doc/")
                self.docs_for(index).pop(doc_id, None)
                return httpx.Response(200, json={"result": "deleted"})

            if method == "PUT" and "/_doc/" in path:
                index, doc_id = path.strip("/").split("/_doc/")
                self.docs_for(index)[doc_id] = _json(request)
                return httpx.Response(201, json={"result": "created"})

            raise AssertionError(f"Unhandled {method} {path}")

        return httpx.MockTransport(handler)


@dataclass(frozen=True)
class _BackendFixture:
    settings: Settings
    client: ElasticClient
    docs: ElasticDocsRepository
    vector: ElasticVectorRepo
    history: ElasticHistoryStorage
    system: ElasticSystemStateStorage
    diagnostics: ElasticHealthDiagnostics
    state: _ElasticTestState


def _build_backend(*, dim: int = 3) -> _BackendFixture:
    settings_obj = Settings(
        persistence_backend="elasticsearch",
        retrieval_mode="dense",
        es_base_url="http://example.test",
    )
    state = _ElasticTestState()
    client = ElasticClient(
        settings_obj=settings_obj,
        client=httpx.Client(base_url="http://example.test", transport=state.transport()),
    )
    docs = ElasticDocsRepository(settings_obj=settings_obj, client=client)
    vector = ElasticVectorRepo(settings_obj=settings_obj, client=client, dim=dim)
    history = ElasticHistoryStorage(settings_obj=settings_obj, client=client)
    system = ElasticSystemStateStorage(settings_obj=settings_obj, client=client)
    diagnostics = ElasticHealthDiagnostics(settings_obj=settings_obj, client=client)
    return _BackendFixture(settings_obj, client, docs, vector, history, system, diagnostics, state)


def test_elastic_client_index_management_and_errors() -> None:
    backend = _build_backend()

    assert backend.client.head_ok(f"/{backend.settings.es_docs_index}") is True
    assert backend.client.head_ok("/missing-index") is False
    assert (
        backend.state.mapping_for(str(backend.settings.es_docs_index))[
            str(backend.settings.es_embedding_field)
        ]["dims"]
        == 3
    )

    backend.state.mapping_for(str(backend.settings.es_docs_index))[
        str(backend.settings.es_embedding_field)
    ] = {"type": "dense_vector", "dims": 9}
    with pytest.raises(ElasticBackendError):
        backend.client.ensure_indices(embed_dim=3)

    error_client = ElasticClient(
        settings_obj=backend.settings,
        client=httpx.Client(
            base_url="http://example.test",
            transport=httpx.MockTransport(lambda request: httpx.Response(500, text="boom")),
        ),
    )
    with pytest.raises(ElasticBackendError):
        error_client.head_ok("/broken")
    with pytest.raises(ElasticBackendError):
        error_client.request_json("GET", "/")

    api_key_settings = backend.settings.model_copy(update={"es_api_key": "secret"})
    assert (
        ElasticClient(
            settings_obj=api_key_settings,
            client=httpx.Client(
                base_url="http://example.test", transport=backend.state.transport()
            ),
        )._resolve_headers()["Authorization"]
        == "ApiKey secret"
    )


def test_elastic_client_tolerates_index_already_exists_race() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(404)
        if request.method == "PUT" and request.url.path.count("/") == 1:
            index = request.url.path.strip("/")
            return httpx.Response(
                400,
                json={
                    "error": {
                        "type": "resource_already_exists_exception",
                        "reason": f"index [{index}] already exists",
                    },
                    "status": 400,
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    client = ElasticClient(
        settings_obj=Settings(
            persistence_backend="elasticsearch",
            search_backend="elasticsearch",
            es_base_url="http://elastic.test",
        ),
        client=httpx.Client(
            base_url="http://elastic.test",
            transport=httpx.MockTransport(handler),
        ),
    )

    client.ensure_indices(embed_dim=None)


def test_document_repository_crud_snapshot_restore_and_scan() -> None:
    backend = _build_backend()
    repo = backend.docs

    stored_ids = repo.store_documents(["alpha", " ", "beta"])
    assert len(stored_ids) == 2

    results, changed, updated_ids = repo.upsert_documents_by_external_id(
        [
            ElasticDocsRepository.UpsertDoc(
                external_id="doc-1",
                content="alpha bravo",
                source_id="src-1",
                metadata={"kind": "faq"},
                chunk_dedup_sha256="dedup-1",
                embedding=[0.1, 0.2, 0.3],
            ),
            ElasticDocsRepository.UpsertDoc(
                external_id="doc-2",
                content="charlie delta",
                source_id="src-2",
                metadata={"kind": "kb"},
            ),
        ]
    )
    assert [row.action for row in results] == ["inserted", "inserted"]
    assert len(changed) == 2
    assert updated_ids == []

    results, changed, updated_ids = repo.upsert_documents_by_external_id(
        [
            ElasticDocsRepository.UpsertDoc(
                external_id="doc-1",
                content="alpha bravo",
                source_id="src-1",
                metadata={"kind": "faq"},
                chunk_dedup_sha256="dedup-1",
            ),
            ElasticDocsRepository.UpsertDoc(
                external_id="doc-2",
                content="charlie delta echo",
                source_id="src-2b",
                metadata={"kind": "kb", "priority": 1},
                chunk_dedup_sha256="dedup-2",
            ),
        ]
    )
    assert [row.action for row in results] == ["unchanged", "updated"]
    assert changed == [(DocId("doc-2"), "charlie delta echo")]
    assert updated_ids == [DocId("doc-2")]

    assert repo.list_ids_by_external_id_prefix("doc-") == [
        (DocId("doc-1"), "doc-1"),
        (DocId("doc-2"), "doc-2"),
    ]
    assert repo.list_ids_by_external_id_prefix(" ") == []

    snapshots_by_id = repo.snapshot_by_ids([DocId("doc-1"), DocId("doc-2")])
    snapshots_by_external_id = repo.snapshot_by_external_ids(["doc-1", "doc-2", "missing"])
    assert len(snapshots_by_id) == 2
    assert len(snapshots_by_external_id) == 2

    deleted_count, deleted_ids, missing, tombstoned = repo.delete_by_external_ids(
        ["doc-2", "missing"]
    )
    assert deleted_count == 1
    assert deleted_ids == [DocId("doc-2")]
    assert missing == ["missing"]
    assert tombstoned == 2
    assert repo.get_tombstoned_external_ids(["doc-2", "missing"]) == {"doc-2", "missing"}
    assert repo.delete_tombstones(["doc-2", "missing"]) == 2
    assert repo.get_tombstoned_external_ids(["doc-2", "missing"]) == set()

    repo.restore_from_snapshots(snapshots_by_external_id)
    restored = repo.get([DocId("doc-1"), DocId("doc-2")])
    assert [doc.external_id for doc in restored] == ["doc-1", "doc-2"]

    repo.delete_documents([DocId("doc-1")])
    assert [doc.external_id for doc in repo.get([DocId("doc-1"), DocId("doc-2")])] == ["doc-2"]

    repo.delete_by_external_ids(["doc-2"])
    repo.hard_delete_by_external_ids(["doc-2"])
    assert repo.get([DocId("doc-2")]) == []
    assert repo.get_tombstoned_external_ids(["doc-2"]) == set()

    with pytest.raises(ValueError):
        repo.upsert_documents_by_external_id(
            [
                ElasticDocsRepository.UpsertDoc(external_id="dup", content="a"),
                ElasticDocsRepository.UpsertDoc(external_id="dup", content="b"),
            ]
        )

    bulk_items = [
        ElasticDocsRepository.UpsertDoc(external_id=f"scan-{idx:03d}", content=f"text {idx}")
        for idx in range(505)
    ]
    repo.upsert_documents_by_external_id(bulk_items)
    all_docs = repo.get_all_documents()
    assert len([doc for doc in all_docs if doc.external_id.startswith("scan-")]) == 505


def test_canonical_import_replaces_large_scope_and_retry_is_idempotent(tmp_path) -> None:
    backend = _build_backend()
    backend.settings.data_dir = tmp_path
    backend.settings.mutation_batch_max_wait_ms = 0
    repo = backend.docs
    ports = DocsMutationPorts(
        build_embedder=lambda: SimpleNamespace(
            embed=lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
        ),
        doc_repo_factory=lambda: repo,
        build_upsert_doc=repo.UpsertDoc,
        vector_repo_factory=lambda **kwargs: backend.vector,
        rebuild_fn=lambda **kwargs: 0,
        write_lock=lambda **kwargs: nullcontext(),
        mutation_journal_factory=lambda: None,
        storage_profile_registry=StorageProfileRegistry(),
    )
    documents = tuple(
        CanonicalImportDocumentInput(external_id=f"doc-{i:05d}", content=f"code {i}")
        for i in range(2503)
    )

    def run(docs, snapshot):
        return execute_import_canonical_sync(
            request=CanonicalImportRequestInput(
                scope="demo", snapshot_id=snapshot, replace_scope=True, documents=docs
            ),
            settings_obj=backend.settings,
            ports=ports,
        )

    assert run(documents, "initial").inserted == 2503
    repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id="foreign", content="preserve other scope", scope="other")]
    )
    replaced = run(documents[:1000], "replacement")
    assert replaced.deleted_sql == replaced.deleted_index == 1503
    assert replaced.deleted_external_ids == [d.external_id for d in documents[1000:]]
    retried = run(documents[:1000], "replacement")
    assert retried.unchanged == 1000
    assert retried.inserted == retried.updated == retried.deleted_sql == retried.deleted_index == 0
    assert len(repo.get_all_documents()) == 1001
    assert backend.vector.ntotal == 1000
    assert repo.get([DocId("foreign")])[0].content == "preserve other scope"
    assert not repo.get_tombstoned_external_ids(replaced.deleted_external_ids)


def test_vector_delete_only_removes_embedding_and_is_idempotent() -> None:
    backend = _build_backend()
    repo = backend.docs
    repo.upsert_documents_by_external_id(
        [
            repo.UpsertDoc(
                external_id="present",
                content="preserve",
                metadata={"keep": True},
                embedding=[1.0, 0.0, 0.0],
            ),
            repo.UpsertDoc(external_id="without-vector", content="also preserve"),
        ]
    )
    before = repo.get([DocId("present"), DocId("without-vector")])
    assert (
        backend.vector.delete(
            [DocId("present"), DocId("missing"), DocId("without-vector"), DocId(" ")]
        )
        == 3
    )
    assert repo.get([DocId("present"), DocId("without-vector")]) == before
    assert backend.vector.ntotal == 0
    assert repo.get([DocId("missing")]) == []
    assert backend.vector.delete([DocId("missing"), DocId("present")]) == 2


def test_vector_delta_allows_only_missing_removal_action() -> None:
    backend = _build_backend()
    backend.docs.store_documents(["unrelated"])
    backend.docs.upsert_documents_by_external_id(
        [backend.docs.UpsertDoc(external_id="present", content="keep")]
    )
    backend.vector.apply_delta_atomic(
        delete_ids=[DocId("missing")], upserts=[(DocId("present"), [1.0, 0.0, 0.0])]
    )
    assert backend.vector.ntotal == 1
    assert backend.docs.get([DocId("missing")]) == []
    # The same ID at a later ordinal is an upsert and must remain strict.
    with pytest.raises(ElasticBackendError, match="update missing: status 404"):
        backend.vector.apply_delta_atomic(
            delete_ids=[DocId("missing")], upserts=[(DocId("missing"), [1.0, 0.0, 0.0])]
        )
    del backend.state.indices[str(backend.settings.es_docs_index)]
    with pytest.raises(ElasticBackendError, match="status 404"):
        backend.vector.delete([DocId("missing")])


def test_vector_history_system_and_diagnostics_roundtrip() -> None:
    backend = _build_backend()
    backend.docs.upsert_documents_by_external_id(
        [
            ElasticDocsRepository.UpsertDoc(external_id="vec-1", content="alpha bravo"),
            ElasticDocsRepository.UpsertDoc(external_id="vec-2", content="alpha charlie"),
            ElasticDocsRepository.UpsertDoc(external_id="vec-3", content="zulu"),
        ]
    )

    backend.vector.upsert(
        [DocId("vec-1"), DocId("vec-2"), DocId("vec-3")],
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    assert backend.vector.ntotal == 3
    assert backend.vector.count_indexed_docs() == 3
    assert backend.vector.get_mapping_dimension() == 3
    assert [doc_id for doc_id, _score in backend.vector.similar([1.0, 0.0, 0.0], 2)] == [
        DocId("vec-1"),
        DocId("vec-2"),
    ]
    assert [doc_id for doc_id, _score in backend.vector.lexical_search("alpha", k=2)] == [
        DocId("vec-1"),
        DocId("vec-2"),
    ]
    assert backend.vector.lexical_search(" ", k=2) == []
    assert backend.vector.similar([1.0, 0.0, 0.0], 0) == []

    assert backend.vector.delete([DocId("vec-3")]) == 1
    assert backend.vector.ntotal == 2
    backend.vector.rebuild([DocId("vec-3")], [[0.0, 1.0, 0.0]])
    assert backend.vector.ntotal == 3

    backend.history.save("q1", "a1", [DocId("vec-1"), DocId("vec-2")])
    backend.history.save("q2", "a2", [DocId("vec-3")])
    entries = backend.history.list_entries(limit=1, offset=0)
    assert len(entries) == 1
    assert entries[0].question == "q2"
    assert entries[0].source_ids == ("vec-3",)

    assert backend.system.get_version("rag-service") == 0
    assert backend.system.bump_version("rag-service") == 1
    assert backend.system.bump_version("rag-service") == 2
    assert backend.system.get_version("rag-service") == 2
    with pytest.raises(ValueError):
        backend.system.get_version(" ")
    with pytest.raises(ValueError):
        backend.system.bump_version("")

    backend.diagnostics.ping_database()
    assert backend.diagnostics.get_documents_count() == 3
    assert backend.diagnostics.get_history_count() == 2
    assert backend.diagnostics.get_document_ids() == ("vec-1", "vec-2", "vec-3")
    assert backend.diagnostics.get_index_ids(id_map_path="ignored") == ("vec-1", "vec-2", "vec-3")
    assert backend.diagnostics.get_retrieval_index_stats(
        index_path="ignored",
        id_map_path="ignored",
        vector_backend="ignored",
    ) == {
        "status": "ok",
        "backend": "elasticsearch",
        "index_path": "rag-docs",
        "id_map_path": "rag-docs",
        "manifest_path": None,
        "dim": 3,
        "vectors": 3,
        "id_map_len": 3,
        "unique_ids": 3,
        "duplicates": 0,
        "documents": 3,
    }
    with TemporaryDirectory() as tmp_dir:
        assert (
            backend.diagnostics.get_incomplete_mutation_records_count(
                coordination_dir=Path(tmp_dir)
            )
            == 0
        )
    assert purge_index_artifacts_noop(index_path="x", id_map_path="y") is None


def test_vector_backend_falls_back_to_zero_for_flat_scores() -> None:
    backend = _build_backend()
    backend.docs.upsert_documents_by_external_id(
        [
            ElasticDocsRepository.UpsertDoc(external_id="flat-1", content="alpha alpha"),
            ElasticDocsRepository.UpsertDoc(external_id="flat-2", content="alpha alpha"),
        ]
    )
    backend.vector.upsert(
        [DocId("flat-1"), DocId("flat-2")],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )

    assert [score for _doc_id, score in backend.vector.similar([1.0, 0.0, 0.0], 2)] == [
        0.0,
        0.0,
    ]
    assert [score for _doc_id, score in backend.vector.lexical_search("alpha", k=2)] == [
        0.0,
        0.0,
    ]


def test_atomic_mutation_executor_handles_unified_elastic_mutation_flow() -> None:
    backend = _build_backend()
    repo = backend.docs
    repo.upsert_documents_by_external_id(
        [
            ElasticDocsRepository.UpsertDoc(external_id="doc-delete", content="delete me"),
            ElasticDocsRepository.UpsertDoc(external_id="doc-tombstone", content="old"),
        ]
    )
    repo.delete_by_external_ids(["doc-tombstone"])

    ports = DocsMutationPorts(
        build_embedder=lambda: None,  # type: ignore[arg-type]
        doc_repo_factory=lambda: repo,
        build_upsert_doc=ElasticDocsRepository.UpsertDoc,
        vector_repo_factory=lambda **kwargs: backend.vector,
        rebuild_fn=lambda **kwargs: 0,
        write_lock=lambda **kwargs: nullcontext(),
        mutation_journal_factory=lambda: None,  # type: ignore[arg-type]
        storage_profile_registry=StorageProfileRegistry(),
        mutation_uow_factory=None,
    )
    executor = AtomicMutationExecutor(settings_obj=backend.settings, ports=ports)
    prepared = PreparedMutation(
        intent=MutationIntent(
            op_id="op-1",
            upserts=(
                MutationUpsertInput(
                    external_id="doc-new",
                    content="brand new",
                    source_id="src-new",
                    metadata={"tier": "gold"},
                ),
                MutationUpsertInput(
                    external_id="doc-tombstone",
                    content="restored",
                    source_id="src-restore",
                ),
            ),
            delete_ids=("doc-delete",),
            delete_external_ids=("missing-doc",),
            source="test",
        ),
        vector_mode_enabled=True,
        precomputed_vectors_by_external_id={
            "doc-new": [1.0, 0.0, 0.0],
            "doc-tombstone": [0.5, 0.5, 0.0],
        },
    )

    summary = executor.execute_locked(prepared=prepared)

    assert summary.op_id == "op-1"
    assert summary.inserted == 2
    assert summary.updated == 0
    assert summary.unchanged == 0
    assert summary.deleted_sql == 1
    assert summary.deleted_index == 1
    assert summary.tombstoned == 1
    assert summary.missing_external_ids == ["missing-doc"]
    assert summary.index_doc_count == 2
    assert [row.external_id for row in list(summary.results or [])] == ["doc-new", "doc-tombstone"]
    assert repo.get_tombstoned_external_ids(["doc-tombstone", "missing-doc"]) == {"missing-doc"}
    docs = repo.get([DocId("doc-new"), DocId("doc-tombstone"), DocId("doc-delete")])
    assert [doc.external_id for doc in docs] == ["doc-new", "doc-tombstone"]
    assert executor.recover_incomplete() == 0
