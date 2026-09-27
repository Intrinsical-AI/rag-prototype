from dataclasses import dataclass

import pytest

from local_rag_backend.core.services.dense_upsert import precompute_vectors


@dataclass
class _Item:
    external_id: str
    content: str


class _Embedder:
    dim = 2

    def __init__(self, vectors=None):
        self.calls = []
        self.vectors = vectors

    def embed(self, texts):
        self.calls.append(list(texts))
        return self.vectors if self.vectors is not None else [[1.0, 0.0] for _ in texts]


def test_precompute_embeds_every_candidate_and_trims_content():
    embedder = _Embedder()
    items = [_Item("same", " stable "), _Item("changed", "new")]
    assert precompute_vectors(items=items, embedder=embedder) == {
        "same": [1.0, 0.0],
        "changed": [1.0, 0.0],
    }
    assert embedder.calls == [["stable", "new"]]
    assert precompute_vectors(items=[], embedder=embedder) == {}
    assert len(embedder.calls) == 1


def test_precompute_rejects_wrong_embedding_count():
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2"):
        precompute_vectors(
            items=[_Item("a", "one"), _Item("b", "two")],
            embedder=_Embedder(vectors=[[1.0, 2.0]]),
        )
