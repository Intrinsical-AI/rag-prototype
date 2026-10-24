"""Exact equivalence of sparse BM25 ranking against the original full-scan algorithm."""

from __future__ import annotations

import random

import numpy as np
import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalRequest
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


class _Repo:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = {doc.id: doc for doc in docs}

    def get(self, ids):
        return [self._docs[doc_id] for doc_id in ids if doc_id in self._docs]


def _legacy_ranking(
    retriever: SparseBM25Retriever, query_tokens: list[str], top_k: int
) -> tuple[list[int], list[float]]:
    """Verbatim ranking from the original retrieve(): full get_scores, full overlap scan."""
    doc_scores = np.asarray(retriever.bm25.get_scores(query_tokens), dtype=np.float32)
    k_eff = min(int(top_k), int(doc_scores.size))
    rank_scores = np.nan_to_num(doc_scores, nan=float("-inf"))
    query_set = set(query_tokens)
    overlaps = [
        sum(token in query_set for token in tokens) for tokens in retriever._tokenized_corpus
    ]
    top_indices = sorted(
        range(len(rank_scores)),
        key=lambda index: (-float(rank_scores[index]), -overlaps[index], index),
    )[:k_eff]
    return [int(i) for i in top_indices], [float(rank_scores[int(i)]) for i in top_indices]


def _random_corpus(rng: random.Random) -> tuple[list[str], list[str]]:
    vocab_size = rng.randint(1, 12)
    vocab = [f"tok{i}" for i in range(vocab_size)]
    n_docs = rng.randint(1, 40)
    documents = []
    for _ in range(n_docs):
        length = rng.choice([0, 0, 1, 2, 3, 5, 8, 13, 30])
        documents.append(" ".join(rng.choice(vocab) for _ in range(length)))
    if not any(documents):
        documents[0] = vocab[0]
    query_vocab = [*vocab, "absent0", "absent1"]
    query = " ".join(rng.choice(query_vocab) for _ in range(rng.randint(1, 6)))
    return documents, [query]


def _build(documents: list[str]) -> SparseBM25Retriever:
    docs = [Document(id=index + 1, content=text) for index, text in enumerate(documents)]
    return SparseBM25Retriever(
        documents=documents,
        doc_ids=[doc.id for doc in docs],
        doc_repo=_Repo(docs),
        preloaded_docs=docs,
    )


@pytest.mark.parametrize("seed", range(200))
def test_ranking_matches_full_scan(seed: int) -> None:
    rng = random.Random(seed)
    documents, queries = _random_corpus(rng)
    retriever = _build(documents)
    for query in queries:
        tokens = SparseBM25Retriever._tokenize(query)
        for top_k in (1, 2, 3, len(documents), len(documents) + 5):
            expected = _legacy_ranking(retriever, tokens, top_k)
            assert retriever._rank(tokens, top_k) == expected


@pytest.mark.parametrize("seed", range(100))
def test_cold_and_warm_postings_match_full_scan(seed: int) -> None:
    """Postings are cached per queried term; cold and warm caches must rank identically."""
    rng = random.Random(30_000 + seed)
    documents, queries = _random_corpus(rng)
    retriever = _build(documents)
    tokens = SparseBM25Retriever._tokenize(queries[0])
    top_k = len(documents) + 5
    expected = _legacy_ranking(retriever, tokens, top_k)
    assert retriever._postings == {}
    assert retriever._rank(tokens, top_k) == expected
    assert set(retriever._postings) == {t for t in tokens if t in retriever.bm25.idf}
    assert retriever._rank(tokens, top_k) == expected


@pytest.mark.parametrize("seed", range(50))
def test_retrieve_matches_full_scan(seed: int) -> None:
    rng = random.Random(10_000 + seed)
    documents, queries = _random_corpus(rng)
    retriever = _build(documents)
    for query in queries:
        tokens = SparseBM25Retriever._tokenize(query)
        top_k = rng.randint(1, len(documents) + 3)
        expected_indices, _ = _legacy_ranking(retriever, tokens, top_k)
        result = retriever.retrieve(RetrievalRequest(query=query, top_k=top_k, mode="sparse"))
        assert [doc.id for doc in result.documents] == [i + 1 for i in expected_indices]


@pytest.mark.parametrize("seed", range(50))
def test_scores_match_rank_bm25_get_scores(seed: int) -> None:
    """Guard against rank-bm25 internals drifting from the inverted-index scorer."""
    rng = random.Random(20_000 + seed)
    documents, queries = _random_corpus(rng)
    retriever = _build(documents)
    for query in queries:
        tokens = SparseBM25Retriever._tokenize(query)
        expected = np.asarray(retriever.bm25.get_scores(tokens))
        assert np.array_equal(retriever._bm25_scores(tokens), expected)


def test_negative_idf_corpus_matches_full_scan() -> None:
    documents = ["common common", "common", "common", ""]
    retriever = _build(documents)
    assert retriever.bm25.idf["common"] < 0
    tokens = SparseBM25Retriever._tokenize("common common")
    for top_k in range(1, len(documents) + 2):
        assert retriever._rank(tokens, top_k) == _legacy_ranking(retriever, tokens, top_k)
    indices, scores = retriever._rank(tokens, len(documents))
    assert indices[0] == 3 and scores[0] == 0


@pytest.mark.parametrize("query", ["shared shared absent", "absent"])
def test_complete_ties_keep_original_document_order(query: str) -> None:
    retriever = _build(["shared", "shared", "shared"])
    tokens = retriever._tokenize(query)
    assert retriever._rank(tokens, 20) == _legacy_ranking(retriever, tokens, 20)
    assert retriever._rank(tokens, 20)[0] == [0, 1, 2]


def test_replaced_model_uses_public_scores_after_cache_warmup() -> None:
    retriever = _build(["alpha", "alpha alpha", "beta"])
    tokens = retriever._tokenize("alpha alpha")
    retriever._rank(tokens, 3)

    class Replacement:
        def get_scores(self, _tokens: list[str]) -> np.ndarray:
            return np.array([0.25, 1.0, 1.0])

    retriever.bm25 = Replacement()
    assert retriever._rank(tokens, 3) == _legacy_ranking(retriever, tokens, 3)
    assert retriever._rank(tokens, 3)[0] == [1, 2, 0]


def test_postings_stay_bounded_by_corpus_vocabulary_and_incidences() -> None:
    retriever = _build(["alpha beta alpha", "beta gamma", "gamma"])
    vocabulary = {term for document in retriever._tokenized_corpus for term in document}
    for term in [*vocabulary, *(f"absent{i}" for i in range(1000))]:
        retriever._rank([term], 3)
    assert set(retriever._postings) == vocabulary
    incidences = sum(len(set(document)) for document in retriever._tokenized_corpus)
    assert (
        sum(indices.nbytes + counts.nbytes for indices, counts in retriever._postings.values())
        <= 16 * incidences
    )
