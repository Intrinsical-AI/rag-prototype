# tests/e2e/test_api.py
import tempfile
from pathlib import Path

from local_rag_backend.http.routers import rag_router


class DummyRagSvc:
    def ask(self, question, top_k=3, *, filters=(), retrieval_mode="sparse"):
        _ = (top_k, filters, retrieval_mode)
        return {
            "answer": f"eco:{question}",
            "docs": [],
            "scores": [],
        }


class DummyRagSvcWithDocs:
    def ask(self, question, top_k=3, *, filters=(), retrieval_mode="sparse"):
        _ = (top_k, filters, retrieval_mode)
        from local_rag_backend.core.domain.entities import Document

        return {
            "answer": f"eco:{question}",
            "docs": [Document(id=42, content="Test document")],
            "scores": [0.95],
        }


# ---------- tests -----------------------------------------------------------
async def test_post_ask_endpoint(asgi_client, monkeypatch):
    async def _override():
        return DummyRagSvc()

    monkeypatch.setattr(rag_router, "get_rag_service", _override, raising=True)
    resp = await asgi_client.post("/api/ask", json={"question": "hola", "k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "eco:hola"
    assert data["sources"] == []


async def test_get_root_frontend_not_found(asgi_client, tmp_path, monkeypatch):
    # Simulate that index.html does not exist neither in package nor in repo
    # 1) Force failure when searching for packaged resources
    monkeypatch.setattr(
        "local_rag_backend.http.main.resources.files",
        lambda *a, **k: object(),  # object without .joinpath -> will raise exception and fallback
    )
    # 2) Fallback points to empty directory
    monkeypatch.setattr("local_rag_backend.http.main.FRONTEND_DIR", tmp_path)
    resp = await asgi_client.get("/")
    assert resp.status_code == 404


async def test_get_root_frontend_packaged_ok(asgi_client, monkeypatch):
    tmpdir = tempfile.TemporaryDirectory()
    idx = Path(tmpdir.name) / "index.html"
    idx.write_text("<!doctype html><html><body>ok</body></html>", encoding="utf-8")

    class _Pkg:
        def joinpath(self, name):
            return idx

    monkeypatch.setattr("local_rag_backend.http.main.resources.files", lambda *_: _Pkg())
    resp = await asgi_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "<body>ok</body>" in resp.text
    tmpdir.cleanup()


async def test_get_frontend_assets_packaged_ok(asgi_client, monkeypatch):
    tmpdir = tempfile.TemporaryDirectory()
    base = Path(tmpdir.name)
    (base / "index.html").write_text(
        "<!doctype html><html><body>ok</body></html>", encoding="utf-8"
    )
    (base / "styles.css").write_text("body{background:#fff}", encoding="utf-8")
    (base / "app.js").write_text("console.log('ok')", encoding="utf-8")

    class _Pkg:
        def joinpath(self, *parts):
            return base.joinpath(*parts)

    monkeypatch.setattr("local_rag_backend.http.main.resources.files", lambda *_: _Pkg())

    r_css = await asgi_client.get("/assets/styles.css")
    assert r_css.status_code == 200
    assert "text/css" in (r_css.headers.get("content-type") or "")
    assert "background" in r_css.text

    r_js = await asgi_client.get("/assets/app.js")
    assert r_js.status_code == 200
    assert "javascript" in (r_js.headers.get("content-type") or "")
    assert "console.log" in r_js.text
    tmpdir.cleanup()


async def test_get_frontend_assets_path_traversal_404(asgi_client):
    r = await asgi_client.get("/assets/../pyproject.toml")
    assert r.status_code == 404


async def test_api_ask_schema_with_sources(asgi_client, monkeypatch):
    # Test API schema validation for /api/ask endpoint with documents
    async def _override():
        return DummyRagSvcWithDocs()

    monkeypatch.setattr(rag_router, "get_rag_service", _override, raising=True)
    resp = await asgi_client.post("/api/ask", json={"question": "test", "k": 1})
    assert resp.status_code == 200
    data = resp.json()

    # Validate response schema
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == 1

    # Validate source schema
    source = data["sources"][0]
    assert "document" in source
    assert "score" in source
    assert "id" in source["document"]
    assert "content" in source["document"]
    assert source["document"]["id"] == "42"
    assert source["document"]["content"] == "Test document"
    assert isinstance(source["score"], int | float)
    assert 0.0 <= source["score"] <= 1.0
