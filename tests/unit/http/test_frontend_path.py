"""Tests for get_frontend_path helper."""

from __future__ import annotations

from pathlib import Path

from local_rag_backend.http.main import get_frontend_path


def test_get_frontend_path_returns_none_when_no_frontend(tmp_path: Path, monkeypatch) -> None:
    import importlib.resources as resources

    import local_rag_backend.http.main as main_mod

    # Make the package resource lookup fail so we fall through to FRONTEND_DIR
    monkeypatch.setattr(resources, "files", lambda _pkg: (_ for _ in ()).throw(AttributeError()))
    monkeypatch.setattr(main_mod, "FRONTEND_DIR", tmp_path)
    result = get_frontend_path()
    # Neither the package resource nor the repo path exists → must return None
    assert result is None


def test_get_frontend_path_returns_path_when_index_html_exists(tmp_path: Path, monkeypatch) -> None:
    import local_rag_backend.http.main as main_mod

    (tmp_path / "index.html").write_text("<html/>", encoding="utf-8")
    monkeypatch.setattr(main_mod, "FRONTEND_DIR", tmp_path)
    result = get_frontend_path()
    assert result is not None
    assert Path(str(result)).name == "index.html"
