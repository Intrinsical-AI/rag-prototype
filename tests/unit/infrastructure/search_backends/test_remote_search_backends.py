from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.infrastructure.search_backends.elastic_like import (
    ElasticLikeSearchRetriever,
)
from local_rag_backend.infrastructure.search_backends.solr import SolrSearchRetriever


class DummyEmbedder:
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            normalized = text.lower()
            if "auth" in normalized:
                out.append([1.0, 0.0])
            elif "sql" in normalized:
                out.append([0.0, 1.0])
            else:
                out.append([0.5, 0.5])
        return out


def test_elastic_like_sparse_builds_filter_query() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_id": "doc-1",
                            "_score": 3.0,
                            "_source": {
                                "external_id": "doc-1",
                                "content": "auth code",
                                "metadata": {"language": "python"},
                            },
                        }
                    ]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://example.test")
    retriever = ElasticLikeSearchRetriever(
        backend_name="opensearch",
        base_url="http://example.test",
        docs_index="rag-docs",
        content_field="content",
        embedding_field="embedding",
        request_timeout_s=5.0,
        verify_tls=False,
        embedder=DummyEmbedder(),
        client=client,
    )
    result = retriever.retrieve(
        RetrievalRequest(
            query="auth",
            top_k=3,
            mode="sparse",
            filters=(RetrievalFilter(field="metadata.language", values=("python",)),),
        )
    )
    assert captured["path"] == "/rag-docs/_search"
    body = captured["body"]
    assert isinstance(body, dict)
    query = body["query"]
    assert isinstance(query, dict)
    bool_query = query["bool"]
    assert isinstance(bool_query, dict)
    assert bool_query["filter"] == [{"terms": {"metadata.language.keyword": ["python"]}}]
    assert [item.document.id for item in result.items] == ["doc-1"]


def test_elastic_like_sparse_supports_metadata_prefixed_filters() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"hits": {"hits": []}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://example.test")
    retriever = ElasticLikeSearchRetriever(
        backend_name="opensearch",
        base_url="http://example.test",
        docs_index="rag-docs",
        content_field="content",
        embedding_field="embedding",
        request_timeout_s=5.0,
        verify_tls=False,
        client=client,
    )

    retriever.retrieve(
        RetrievalRequest(
            query="sensor",
            top_k=3,
            mode="sparse",
            filters=(
                RetrievalFilter(field="metadata.doc_type", values=("net_node",)),
                RetrievalFilter(field="metadata.sensor_id", values=("sensor-a",)),
            ),
        )
    )

    body = captured["body"]
    assert isinstance(body, dict)
    query = body["query"]
    assert isinstance(query, dict)
    bool_query = query["bool"]
    assert isinstance(bool_query, dict)
    assert bool_query["filter"] == [
        {"terms": {"metadata.doc_type.keyword": ["net_node"]}},
        {"terms": {"metadata.sensor_id.keyword": ["sensor-a"]}},
    ]


def test_solr_rejects_dense_mode() -> None:
    retriever = SolrSearchRetriever(
        base_url="http://example.test",
        core="rag-docs",
        content_field="content",
        request_timeout_s=5.0,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )
    with pytest.raises(ValueError, match="supports only retrieval_mode=sparse"):
        retriever.retrieve(RetrievalRequest(query="auth", top_k=3, mode="dense"))


def test_elastic_like_dense_builds_knn_query_and_applies_min_score() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_id": "doc-1",
                            "_score": 10.0,
                            "_source": {"external_id": "doc-1", "content": "auth code"},
                        },
                        {
                            "_id": "doc-2",
                            "_score": 5.0,
                            "_source": {"external_id": "doc-2", "content": "sql code"},
                        },
                    ]
                }
            },
        )

    retriever = ElasticLikeSearchRetriever(
        backend_name="opensearch",
        base_url="http://example.test",
        docs_index="rag-docs",
        content_field="content",
        embedding_field="embedding",
        request_timeout_s=5.0,
        verify_tls=False,
        embedder=DummyEmbedder(),
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://example.test"),
    )

    result = retriever.retrieve(
        RetrievalRequest(
            query="auth",
            top_k=2,
            candidate_k=1,
            mode="dense",
            min_score=0.5,
            filters=(RetrievalFilter(field="metadata.language", values=("python",)),),
        )
    )

    body = captured["body"]
    assert isinstance(body, dict)
    knn = body["knn"]
    assert isinstance(knn, dict)
    assert knn["k"] == 2
    assert knn["num_candidates"] == 2
    assert knn["filter"] == {
        "bool": {"filter": [{"terms": {"metadata.language.keyword": ["python"]}}]}
    }
    assert [item.document.id for item in result.items] == ["doc-1"]
    assert result.mode_used == "dense"


def test_elastic_like_dual_uses_dual_candidate_floor_and_reranks() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_id": "doc-auth",
                            "_score": 9.0,
                            "_source": {"external_id": "doc-auth", "content": "auth code"},
                        },
                        {
                            "_id": "doc-sql",
                            "_score": 6.0,
                            "_source": {"external_id": "doc-sql", "content": "sql code"},
                        },
                    ]
                }
            },
        )

    retriever = ElasticLikeSearchRetriever(
        backend_name="opensearch",
        base_url="http://example.test",
        docs_index="rag-docs",
        content_field="content",
        embedding_field="embedding",
        request_timeout_s=5.0,
        verify_tls=False,
        embedder=DummyEmbedder(),
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://example.test"),
    )

    result = retriever.retrieve(
        RetrievalRequest(query="auth", top_k=2, mode="dual", dual_candidate_k=1)
    )

    assert requests[0]["size"] == 2
    assert [item.document.id for item in result.items] == ["doc-auth", "doc-sql"]
    assert result.mode_used == "dual"


@pytest.mark.parametrize("status_code", [404, 503])
def test_elastic_like_propagates_http_errors(status_code):
    retriever = ElasticLikeSearchRetriever(
        backend_name="opensearch",
        base_url="http://example.test",
        docs_index="rag-docs",
        content_field="content",
        embedding_field="embedding",
        request_timeout_s=5.0,
        verify_tls=False,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status_code, json={"error": "boom"})
            ),
            base_url="http://example.test",
        ),
    )

    with pytest.raises(httpx.HTTPStatusError):
        retriever.retrieve(RetrievalRequest(query="auth", top_k=1, mode="sparse"))


@pytest.mark.parametrize("mode", ["dense", "dual"])
def test_elastic_like_requires_embedder_for_dense_modes(mode):
    retriever = ElasticLikeSearchRetriever(
        backend_name="opensearch",
        base_url="http://example.test",
        docs_index="rag-docs",
        content_field="content",
        embedding_field="embedding",
        request_timeout_s=5.0,
        verify_tls=False,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )

    with pytest.raises(RuntimeError, match="requires an embedder"):
        retriever.retrieve(RetrievalRequest(query="auth", top_k=1, mode=mode))


def test_elastic_like_client_configuration_uses_api_key_and_basic_auth(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.search_backends.elastic_like.httpx.Client", DummyClient
    )

    retriever = ElasticLikeSearchRetriever(
        backend_name="opensearch",
        base_url="https://example.test",
        docs_index="rag-docs",
        content_field="content",
        embedding_field="embedding",
        request_timeout_s=7.5,
        verify_tls=False,
        api_key="secret",
        username="user",
        password="pass",
    )

    assert retriever._owns_client is True
    assert captured["base_url"] == "https://example.test"
    assert captured["verify"] is False
    assert captured["timeout"] == 7.5
    assert captured["auth"] == ("user", "pass")
    assert captured["headers"]["Authorization"] == "ApiKey secret"


def test_solr_builds_filter_queries_and_parses_metadata_json() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["filters"] = request.url.params.get_list("fq")
        return httpx.Response(
            200,
            json={
                "response": {
                    "docs": [
                        {
                            "id": "doc-1",
                            "content": "auth code",
                            "metadata_json": json.dumps({"language": "python"}),
                            "scope": "repo",
                            "score": 3.0,
                        }
                    ]
                }
            },
        )

    retriever = SolrSearchRetriever(
        base_url="http://example.test",
        core="rag-docs",
        content_field="content",
        request_timeout_s=5.0,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://example.test"),
    )

    result = retriever.retrieve(
        RetrievalRequest(
            query="auth",
            top_k=3,
            mode="sparse",
            filters=(
                RetrievalFilter(field="scope", values=("repo",)),
                RetrievalFilter(field="metadata.language", values=("python", "rust")),
            ),
        )
    )

    assert captured["filters"] == ['scope:"repo"', 'metadata.language:("python" OR "rust")']
    assert result.items[0].document.metadata == {"language": "python", "scope": "repo"}


def test_solr_supports_metadata_prefixed_filter_fields() -> None:
    retriever = SolrSearchRetriever(
        base_url="http://example.test",
        core="rag-docs",
        content_field="content",
        request_timeout_s=5.0,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )

    assert (
        retriever._fq(RetrievalFilter(field="metadata.doc_type", values=("net_node",)))
        == 'metadata.doc_type:"net_node"'
    )


def test_solr_fq_formats_single_value() -> None:
    retriever = SolrSearchRetriever(
        base_url="http://example.test",
        core="rag-docs",
        content_field="content",
        request_timeout_s=5.0,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )

    assert retriever._fq(RetrievalFilter(field="scope", values=("repo",))) == 'scope:"repo"'


@pytest.mark.parametrize("status_code", [400, 500])
def test_solr_propagates_http_errors(status_code):
    retriever = SolrSearchRetriever(
        base_url="http://example.test",
        core="rag-docs",
        content_field="content",
        request_timeout_s=5.0,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status_code, json={"error": "boom"})
            ),
            base_url="http://example.test",
        ),
    )

    with pytest.raises(httpx.HTTPStatusError):
        retriever.retrieve(RetrievalRequest(query="auth", top_k=3, mode="sparse"))
