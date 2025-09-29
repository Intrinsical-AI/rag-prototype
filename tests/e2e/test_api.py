# tests/e2e/test_api.py
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.app.main import app


class DummyRagSvc:
    def ask(self, question, top_k=3):
        return {
            "answer": f"eco:{question}",
            "docs": [],
            "scores": [],
        }


class DummyRagSvcWithDocs:
    def ask(self, question, top_k=3):
        from local_rag_backend.core.domain.entities import Document

        return {
            "answer": f"eco:{question}",
            "docs": [Document(id=42, content="Test document")],
            "scores": [0.95],
        }


# ---------- override dependency --------------------------------------------
app.dependency_overrides = {}
app.dependency_overrides[get_rag_service] = lambda: DummyRagSvc()

client = TestClient(app)


# ---------- tests -----------------------------------------------------------
def test_post_ask_endpoint():
    resp = client.post("/api/ask", json={"question": "hola", "k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "eco:hola"
    assert data["sources"] == []


def test_get_root_frontend_not_found(tmp_path, monkeypatch):
    # Simulate that index.html does not exist neither in package nor in repo
    # 1) Force failure when searching for packaged resources
    monkeypatch.setattr(
        "local_rag_backend.app.main.resources.files",
        lambda *a, **k: object(),  # object without .joinpath -> will raise exception and fallback
    )
    # 2) Fallback points to empty directory
    monkeypatch.setattr("local_rag_backend.app.main.FRONTEND_DIR", tmp_path)
    resp = client.get("/")
    assert resp.status_code == 404


def test_get_root_frontend_packaged_ok(monkeypatch):
    tmpdir = tempfile.TemporaryDirectory()
    idx = Path(tmpdir.name) / "index.html"
    idx.write_text("<!doctype html><html><body>ok</body></html>", encoding="utf-8")

    class _Pkg:
        def joinpath(self, name):
            return idx

    monkeypatch.setattr("local_rag_backend.app.main.resources.files", lambda *_: _Pkg())
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "<body>ok</body>" in resp.text
    tmpdir.cleanup()


def test_api_ask_schema_with_sources():
    # Test API schema validation for /api/ask endpoint with documents
    app.dependency_overrides[get_rag_service] = lambda: DummyRagSvcWithDocs()
    client_with_docs = TestClient(app)

    resp = client_with_docs.post("/api/ask", json={"question": "test", "k": 1})
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
    assert source["document"]["id"] == 42
    assert source["document"]["content"] == "Test document"
    assert isinstance(source["score"], (int, float))
    assert 0.0 <= source["score"] <= 1.0
