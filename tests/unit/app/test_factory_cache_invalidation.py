from types import SimpleNamespace

import pytest

from local_rag_backend.app import factory


@pytest.mark.unit
async def test_get_rag_service_cache_is_invalidated_by_reload_token(tmp_path, monkeypatch):
    """
    Regression test: in multi-worker deployments, `reset_rag_service()` only cleared
    the in-process cache, leaving other workers serving stale retrievers/generators.
    """
    monkeypatch.setattr(factory.settings, "data_dir", tmp_path, raising=False)

    # Ensure a clean slate for this module-level cache.
    factory._get_cached_rag_service.cache_clear()
    token_path = tmp_path / ".rag_service_reload_token"
    token_path.unlink(missing_ok=True)

    built: list[object] = []

    def _build():
        obj = object()
        built.append(obj)
        return obj

    monkeypatch.setattr(factory, "build_rag_service", _build, raising=True)

    svc1 = await factory.get_rag_service()
    svc2 = await factory.get_rag_service()
    assert svc1 is svc2
    assert built == [svc1]

    # Simulate an external invalidation (another process updated the token file).
    token_path.write_text("new-token", encoding="utf-8")

    svc3 = await factory.get_rag_service()
    assert svc3 is not svc1
    assert built == [svc1, svc3]

    svc4 = await factory.get_rag_service()
    assert svc4 is svc3


def test_write_reload_token_uses_unique_tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(factory.settings, "data_dir", tmp_path, raising=False)
    token_path = tmp_path / ".rag_service_reload_token"

    seen_sources: set[str] = set()
    real_replace = factory.os.replace

    def _replace(src, dst):
        src_s = str(src)
        if src_s in seen_sources:
            raise FileNotFoundError("duplicate temporary path")
        seen_sources.add(src_s)
        return real_replace(src, dst)

    monkeypatch.setattr(factory.os, "replace", _replace, raising=True)

    factory._write_reload_token("v1")
    factory._write_reload_token("v2")

    assert token_path.read_text(encoding="utf-8") == "v2"


def test_reload_token_path_uses_coordination_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        factory,
        "settings",
        SimpleNamespace(get_coordination_dir=lambda: tmp_path),
        raising=True,
    )
    assert factory._reload_token_path() == tmp_path / ".rag_service_reload_token"
