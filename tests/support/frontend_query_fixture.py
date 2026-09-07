"""Produce real accepted query DTOs for the local Chromium regression smoke.

Only Elasticsearch HTTP is simulated. Models, storage mapping, read adapter and
HTTP DTO conversion are the application implementations; no private data is read.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from local_rag_backend.composition.adapters import _RepoDocsReadPort
from local_rag_backend.core.services.canonical_import_transport import (
    validate_canonical_import_payload,
)
from local_rag_backend.http.routers.docs import _to_document_in_db
from local_rag_backend.http.schemas.docs import DocsMutateRequest
from local_rag_backend.infrastructure.persistence.elasticsearch.client import ElasticClient
from local_rag_backend.infrastructure.persistence.elasticsearch.document_storage import (
    ElasticDocsRepository,
)
from local_rag_backend.settings import Settings

PROBE_ID = (
    "probe<img src='/missing-probe' "
    "onerror=\"document.documentElement.dataset.domProbe='executed'\">"
)


def accepted_query() -> list[dict[str, Any]]:
    documents = [
        {"external_id": PROBE_ID, "content": "Preview: <b>literal HTML</b>"},
        {"external_id": "doc:0198c3f7-3789-7123-89ab-0123456789ab", "content": "UUID document"},
        {"external_id": "repogpt:demo:src/example.py:function:hello", "content": "Canonical code"},
    ]
    canonical = validate_canonical_import_payload(
        {"scope": "browser-smoke", "snapshot_id": "one", "documents": documents}
    )
    mutation = DocsMutateRequest.model_validate({"upserts": documents})
    assert mutation.upserts[0].external_id == PROBE_ID
    state: dict[str, dict[str, Any]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200)
        if request.url.path.endswith("/_mapping"):
            index = request.url.path.split("/")[1]
            return httpx.Response(
                200, json={index: {"mappings": {"properties": {"scope": {}, "snapshot_id": {}}}}}
            )
        body = json.loads(request.content) if request.url.path != "/_bulk" else None
        if request.url.path.endswith("/_mget"):
            return httpx.Response(
                200,
                json={
                    "docs": [
                        {"_id": id_, "found": id_ in state, "_source": state.get(id_)}
                        for id_ in body["ids"]
                    ]
                },
            )
        if request.url.path == "/_bulk":
            ops = [json.loads(line) for line in request.content.decode().splitlines()]
            items = []
            for action, source in zip(ops[::2], ops[1::2], strict=True):
                id_ = action["index"]["_id"]
                state[id_] = source
                items.append({"index": {"_id": id_, "status": 201}})
            return httpx.Response(200, json={"errors": False, "items": items})
        if request.url.path.endswith("/_search"):
            after = body.get("search_after", [""])[0]
            ids = [id_ for id_ in sorted(state) if id_ > after][: body["size"]]
            return httpx.Response(
                200,
                json={
                    "hits": {
                        "hits": [{"_id": id_, "_source": state[id_], "sort": [id_]} for id_ in ids]
                    }
                },
            )
        raise AssertionError((request.method, request.url.path))

    settings = Settings(
        persistence_backend="elasticsearch", retrieval_mode="dense", es_base_url="http://smoke.test"
    )
    with httpx.Client(base_url="http://smoke.test", transport=httpx.MockTransport(handler)) as http:
        client = ElasticClient(settings_obj=settings, client=http)
        repo = ElasticDocsRepository(settings_obj=settings, client=client)
        repo.upsert_documents_by_external_id(
            [
                repo.UpsertDoc(external_id=d.external_id, content=d.content)
                for d in canonical.documents
            ]
        )
        reader = _RepoDocsReadPort(doc_repo_factory=lambda: repo)
        output = [
            _to_document_in_db(row).model_dump()
            for row in reader.query_docs(limit=100, offset=0, filters=())
        ]
    assert PROBE_ID in state
    assert {d["id"] for d in output} == {d.external_id for d in canonical.documents}
    return output


if __name__ == "__main__":
    print(json.dumps(accepted_query()))
