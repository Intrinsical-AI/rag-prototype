from __future__ import annotations

from local_rag_backend.http.routers import rag_router
from local_rag_backend.settings import get_settings


async def test_ask_uses_configured_dual_candidate_pool(
    asgi_client, in_memory_sqlite, monkeypatch
) -> None:
    _ = in_memory_sqlite
    settings = get_settings()
    monkeypatch.setattr(settings, "retrieval_mode", "dual")
    monkeypatch.setattr(settings, "dual_candidate_k", 7)
    seen: dict[str, object] = {}

    class _Service:
        def ask(
            self,
            question: str,
            top_k: int = 3,
            *,
            filters=(),
            dual_candidate_k: int | None = None,
            retrieval_mode: str = "sparse",
        ):
            seen.update(
                question=question,
                top_k=top_k,
                filters=filters,
                dual_candidate_k=dual_candidate_k,
                retrieval_mode=retrieval_mode,
            )
            return {"answer": "ok", "docs": [], "scores": []}

    async def _service() -> _Service:
        return _Service()

    monkeypatch.setattr(rag_router, "get_rag_service", _service)
    response = await asgi_client.post("/api/ask", json={"question": "auth", "k": 1})

    assert response.status_code == 200
    assert seen["retrieval_mode"] == "dual"
    assert seen["dual_candidate_k"] == 7
