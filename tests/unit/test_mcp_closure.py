from __future__ import annotations

import anyio
import httpx
import pytest
from mcp import McpError, types
from openai import OpenAI

from local_rag_backend import mcp_server
from local_rag_backend.composition.container import AppContainer
from local_rag_backend.core.errors import LLMResponseError
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.infrastructure.llms import openai_chat
from local_rag_backend.infrastructure.persistence.read_snapshot import SnapshotError
from local_rag_backend.settings import Settings


def invoke(name, arguments=None):
    async def run():
        result = await mcp_server.create_server().request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(name=name, arguments=arguments)
            )
        )
        return result.root

    return anyio.run(run)


@pytest.mark.parametrize("value", [None, "3", -1, True, 0.0])
def test_mcp_invalid_failure_count_is_a_refusal_before_mutation(value, monkeypatch):
    def forbidden():
        pytest.fail("Malformed import reached mutable container")

    monkeypatch.setattr(mcp_server, "get_cli_container", forbidden)
    result = invoke(
        "rag_import_canonical",
        {
            "payload": {
                "scope": "test",
                "snapshot_id": "test",
                "stats": {"error_count": value},
                "documents": [{"external_id": "one", "content": "synthetic"}],
            }
        },
    )
    assert result.isError
    assert result.structuredContent["error"]["code"] == "ToolOperationError"


@pytest.mark.parametrize("value", [None, "3", -1, True, 0.0])
async def test_http_invalid_failure_count_refuses_before_mutation(value, monkeypatch, asgi_client):
    def forbidden(*args, **kwargs):
        pytest.fail("Malformed import constructed mutation services")

    monkeypatch.setattr(AppContainer, "build_docs_mutation_bundle", forbidden)
    response = await asgi_client.post(
        "/api/docs/import-canonical",
        json={
            "scope": "test",
            "snapshot_id": "test",
            "stats": {"error_count": value},
            "documents": [{"external_id": "one", "content": "synthetic"}],
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize("k", [0, 101, True, 1.0])
def test_eval_bounds_reject_before_runtime_loading(k, monkeypatch):
    def forbidden():
        pytest.fail("Invalid eval bound loaded runtime")

    monkeypatch.setattr(mcp_server, "get_cli_runtime_snapshot", forbidden)
    with pytest.raises(McpError) as caught:
        invoke("rag_eval", {"k": k})
    assert caught.value.error.code == -32602


def test_ephemeral_provider_does_not_retry_a_real_sdk_failure(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "synthetic unavailable"}})

    client = OpenAI(
        api_key="synthetic-only",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(openai_chat, "create_openai_client", lambda **kwargs: client)
    generator = openai_chat.OpenAIGenerator(
        settings_obj=Settings(openai_api_key="synthetic-only"), max_retries=0
    )
    with pytest.raises(LLMResponseError):
        generator.generate("synthetic", ["synthetic"])
    assert len(requests) == 1
    client.close()


def test_ephemeral_builder_passes_zero_retries_without_changing_native_defaults(monkeypatch):
    settings = Settings(openai_api_key="synthetic-only")
    embedder = OpenAIEmbedder(settings_obj=settings, max_retries=0)
    native = OpenAIEmbedder(settings_obj=settings)
    try:
        assert embedder.client.max_retries == 0
        assert native.client.max_retries == 2
    finally:
        embedder.client.close()
        native.client.close()


def test_snapshot_refusal_does_not_leak_private_provider_details(monkeypatch):
    def refused(settings):
        raise SnapshotError("PRIVATE-SNAPSHOT-CONTENT")

    monkeypatch.setattr(mcp_server, "build_ephemeral_service", refused)
    result = invoke("rag_ask_ephemeral", {"question": "synthetic"})
    assert result.isError
    assert "PRIVATE-SNAPSHOT-CONTENT" not in result.model_dump_json()
