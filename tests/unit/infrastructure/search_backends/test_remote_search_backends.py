import json

import httpx
import pytest

from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.infrastructure.search_backends.elastic_like import (
    ElasticLikeSearchRetriever,
)
from local_rag_backend.infrastructure.search_backends.solr import SolrSearchRetriever


class DummyEmbedder:
    dim = 2

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_elastic_like_sparse_builds_filter_query():
    captured = {}

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
            filters=(RetrievalFilter(field="language", values=("python",)),),
        )
    )
    assert captured["path"] == "/rag-docs/_search"
    bool_query = captured["body"]["query"]["bool"]
    assert bool_query["filter"] == [{"terms": {"metadata.language.keyword": ["python"]}}]
    assert [item.document.id for item in result.items] == ["doc-1"]


def test_solr_rejects_dense_mode():
    retriever = SolrSearchRetriever(
        base_url="http://example.test",
        core="rag-docs",
        content_field="content",
        request_timeout_s=5.0,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )
    with pytest.raises(ValueError, match="supports only retrieval_mode=sparse"):
        retriever.retrieve(RetrievalRequest(query="auth", top_k=3, mode="dense"))
