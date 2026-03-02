from __future__ import annotations

import hashlib

import pytest

from local_rag_backend.core.services.dense_upsert import (
    precompute_vectors_for_changed_items,
    sync_dense_after_upsert,
)


class _Item:
    def __init__(self, external_id: str, content: str) -> None:
        self.external_id = external_id
        self.content = content


class _State:
    def __init__(self, content: str, content_sha256: str | None) -> None:
        self.content = content
        self.content_sha256 = content_sha256


class _Result:
    def __init__(self, external_id: str, id: int, content_changed: bool) -> None:
        self.external_id = external_id
        self.id = id
        self.content_changed = content_changed


class _DocRepoExisting:
    def __init__(self, by_external_id: dict[str, _State]) -> None:
        self._by_external_id = by_external_id

    def get_existing_doc_states_by_external_id(self, external_ids):
        return {k: v for k, v in self._by_external_id.items() if k in set(external_ids)}


class _Embedder:
    dim = 2

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._vectors = vectors

    def embed(self, texts):
        texts_list = [str(t) for t in texts]
        self.calls.append(texts_list)
        if self._vectors is not None:
            return self._vectors
        return [[float(i), float(i + 1)] for i, _ in enumerate(texts_list)]


class _VecRepo:
    def __init__(self, *, fail_upsert: bool = False) -> None:
        self.delete_calls: list[list[int]] = []
        self.upsert_calls: list[tuple[list[int], list[list[float]]]] = []
        self.fail_upsert = fail_upsert

    def delete(self, ids):
        self.delete_calls.append([int(i) for i in ids])
        return len(ids)

    def upsert(self, ids, vectors):
        if self.fail_upsert:
            raise RuntimeError("upsert-fail")
        self.upsert_calls.append(([int(i) for i in ids], [list(v) for v in vectors]))


def test_precompute_vectors_for_changed_and_new_items_only() -> None:
    stable_sha = hashlib.sha256(b"stable").hexdigest()
    repo = _DocRepoExisting(
        {
            "same": _State(content="stable", content_sha256=stable_sha),
            # Same content but missing hash: we intentionally re-embed to harden old data.
            "rehash-needed": _State(content="rehash", content_sha256=None),
            "changed": _State(content="old", content_sha256=hashlib.sha256(b"old").hexdigest()),
        }
    )
    embedder = _Embedder()
    items = [
        _Item("same", " stable "),
        _Item("rehash-needed", " rehash "),
        _Item("changed", "new"),
        _Item("new", "brand new"),
    ]

    out = precompute_vectors_for_changed_items(items=items, doc_repo=repo, embedder=embedder)

    assert list(out.keys()) == ["same", "rehash-needed", "changed", "new"]
    assert embedder.calls == [["stable", "rehash", "new", "brand new"]]


def test_precompute_vectors_raises_when_embedder_returns_wrong_count() -> None:
    repo = _DocRepoExisting({})
    embedder = _Embedder(vectors=[[1.0, 2.0]])  # only one vector for two inputs
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2"):
        precompute_vectors_for_changed_items(
            items=[_Item("a", "one"), _Item("b", "two")],
            doc_repo=repo,
            embedder=embedder,
        )


def test_sync_dense_after_upsert_happy_path() -> None:
    vec = _VecRepo()
    rebuilt = sync_dense_after_upsert(
        results=[_Result("a", 11, True), _Result("b", 12, False), _Result("c", 13, True)],
        updated_content_ids=[99],
        vectors_by_external_id={"a": [1.0, 2.0], "c": [3.0, 4.0]},
        vec_repo=vec,
        doc_repo=object(),  # not used on success path
        embedder=object(),  # not used on success path
    )

    assert rebuilt is False
    assert vec.delete_calls == [[99]]
    assert vec.upsert_calls == [([11, 13], [[1.0, 2.0], [3.0, 4.0]])]


def test_sync_dense_after_upsert_fails_fast_when_vectors_missing() -> None:
    vec = _VecRepo()
    with pytest.raises(RuntimeError, match="Missing precomputed vectors"):
        sync_dense_after_upsert(
            results=[_Result("a", 1, True)],
            updated_content_ids=[],
            vectors_by_external_id={},
            vec_repo=vec,
            doc_repo=object(),
            embedder=object(),
        )


def test_sync_dense_after_upsert_fallback_rebuild() -> None:
    vec = _VecRepo(fail_upsert=True)
    rebuild_calls: list[tuple[object, object, object]] = []

    def _rebuild(*, doc_repo, vec_repo, embedder):
        rebuild_calls.append((doc_repo, vec_repo, embedder))
        return 7

    doc_repo = object()
    embedder = object()
    rebuilt = sync_dense_after_upsert(
        results=[_Result("a", 1, True)],
        updated_content_ids=[1],
        vectors_by_external_id={"a": [1.0, 2.0]},
        vec_repo=vec,
        doc_repo=doc_repo,
        embedder=embedder,
        rebuild_fn=_rebuild,
    )

    assert rebuilt is True
    assert vec.delete_calls == [[1]]
    assert len(rebuild_calls) == 1
    assert rebuild_calls[0][0] is doc_repo
    assert rebuild_calls[0][1] is vec
    assert rebuild_calls[0][2] is embedder
