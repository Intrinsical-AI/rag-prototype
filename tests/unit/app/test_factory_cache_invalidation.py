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
