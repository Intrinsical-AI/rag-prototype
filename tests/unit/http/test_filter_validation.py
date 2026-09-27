import pytest


@pytest.mark.parametrize("values", ["demo", [], [" "], ["", "demo"], [1], [None], {"demo": True}])
@pytest.mark.parametrize("path", ["/api/docs/query", "/api/ask", "/api/ask_eval"])
async def test_invalid_filter_values_are_rejected_before_runtime(asgi_client, path, values):
    filters = [{"field": "scope", "values": values}]
    payload = {"filters": filters}
    if path == "/api/ask":
        payload["question"] = "hello"
    elif path == "/api/ask_eval":
        payload = {"question": "hello", "config": {"retrieval_mode": "sparse", "filters": filters}}
    response = await asgi_client.post(path, json=payload)
    assert response.status_code == 422


def test_http_filter_values_trim():
    from local_rag_backend.http.schemas.rag_api_models import RetrievalFilterModel

    item = RetrievalFilterModel(field="scope", values=[" demo "])
    assert item.values == ["demo"]
    assert item.to_domain().values == ("demo",)
