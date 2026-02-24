import pytest

from local_rag_backend.settings import settings


@pytest.mark.unit
async def test_ask_rejects_overlong_question(asgi_client, in_memory_sqlite, monkeypatch):
    # FastAPI may resolve dependencies even when request validation fails; ensure
    # the RAG service can be constructed so we reliably assert schema behavior.
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)
    r = await asgi_client.post("/api/ask", json={"question": "x" * 5000, "k": 3})
    assert r.status_code == 422


@pytest.mark.unit
async def test_docs_rejects_too_many_texts(asgi_client, in_memory_sqlite):
    r = await asgi_client.post("/api/docs", json={"texts": ["a"] * 65})
    assert r.status_code == 422


@pytest.mark.unit
async def test_docs_rejects_overlong_text_item(asgi_client, in_memory_sqlite):
    r = await asgi_client.post("/api/docs", json={"texts": ["x" * 20001]})
    assert r.status_code == 422


@pytest.mark.unit
async def test_openrouter_generate_rejects_overlong_fields(asgi_client, in_memory_sqlite):
    r = await asgi_client.post(
        "/api/openrouter/generate",
        json={
            "system_instruction": "x" * 9000,
            "user_content": "hi",
        },
    )
    assert r.status_code == 422


@pytest.mark.unit
async def test_openrouter_generate_rejects_invalid_sampling_params(asgi_client, in_memory_sqlite):
    r1 = await asgi_client.post(
        "/api/openrouter/generate",
        json={
            "system_instruction": "sys",
            "user_content": "hi",
            "top_p": 1.5,
        },
    )
    assert r1.status_code == 422

    r2 = await asgi_client.post(
        "/api/openrouter/generate",
        json={
            "system_instruction": "sys",
            "user_content": "hi",
            "temperature": -0.1,
        },
    )
    assert r2.status_code == 422

    r3 = await asgi_client.post(
        "/api/openrouter/generate",
        json={
            "system_instruction": "sys",
            "user_content": "hi",
            "max_tokens": 0,
        },
    )
    assert r3.status_code == 422


@pytest.mark.unit
async def test_delete_by_external_id_rejects_overlong_values(asgi_client, in_memory_sqlite):
    r = await asgi_client.post(
        "/api/docs/delete_by_external_id",
        json={"external_ids": ["x" * 513]},
    )
    assert r.status_code == 422
