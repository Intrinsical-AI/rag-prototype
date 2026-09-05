import pytest

from local_rag_backend.core.use_cases.errors import NotFoundError
from local_rag_backend.http import main


@pytest.mark.parametrize(
    "asset_path",
    ["../secret.txt", "nested/../../secret.txt", r"..\secret.txt", "/secret.txt", "./app.js"],
)
def test_frontend_asset_rejects_unsafe_paths(asset_path):
    with pytest.raises(NotFoundError):
        main._get_frontend_asset(asset_path)


def test_frontend_asset_allows_nested_file(tmp_path, monkeypatch):
    root = tmp_path / "frontend"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "app.js").write_text("console.log('ok')", encoding="utf-8")
    monkeypatch.setattr(main, "FRONTEND_DIR", root)

    data, media_type = main._get_frontend_asset("nested/app.js")

    assert data == b"console.log('ok')"
    assert "javascript" in media_type


@pytest.mark.parametrize("packaged", [False, True])
def test_frontend_asset_rejects_symlink_escape(tmp_path, monkeypatch, packaged):
    root = tmp_path / "frontend"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    link = root / "escape.txt"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform")
    monkeypatch.setattr(main, "FRONTEND_DIR", root)
    if packaged:
        monkeypatch.setattr(main.resources, "files", lambda package: root)

    with pytest.raises(NotFoundError):
        main._get_frontend_asset("escape.txt")
