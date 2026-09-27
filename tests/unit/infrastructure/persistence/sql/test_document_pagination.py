import pytest
from sqlalchemy import event

from local_rag_backend.core.domain.retrieval import RetrievalFilter
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage


def test_sql_pagination_filters_before_limit_with_unique_stable_order(in_memory_sqlite):
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    repo.upsert_documents_by_external_id(
        [
            repo.UpsertDoc(
                external_id=f"doc:{index}",
                content=f"document {index}",
                source_id="source",
                scope="target" if index % 2 == 0 else "other",
                metadata={"nested": {"tags": ["red", "blue"]}, "flag": True},
            )
            for index in range(15)
        ]
    )
    statements = []
    event.listen(
        in_memory_sqlite.kw["bind"],
        "before_cursor_execute",
        lambda _c, _cur, sql, _p, _ctx, _many: statements.append(sql),
    )
    filters = (
        RetrievalFilter("scope", ("target",)),
        RetrievalFilter("metadata.nested.tags", ("blue",)),
        RetrievalFilter("metadata.flag", ("True",)),
    )
    first = repo.query_documents(limit=3, offset=0, filters=filters)
    second = repo.query_documents(limit=3, offset=3, filters=filters)
    tail = repo.query_documents(limit=3, offset=6, filters=filters)
    assert [len(first), len(second), len(tail)] == [3, 3, 2]
    ids = [doc.id for doc in first + second + tail]
    assert ids == sorted(set(ids))
    assert all(doc.metadata["scope"] == "target" for doc in first + second + tail)
    assert all("ORDER BY documents.doc_id" in sql and "LIMIT" in sql for sql in statements)


@pytest.mark.parametrize(
    "value,expected",
    [(" padded ", "padded"), ({"key": "value"}, "{'key': 'value'}"), (["red", "blue"], "blue")],
)
def test_metadata_pagination_uses_same_semantics_as_retrieval(in_memory_sqlite, value, expected):
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id="doc", content="content", metadata={"value": value})]
    )
    assert [
        doc.external_id
        for doc in repo.query_documents(
            limit=1, offset=0, filters=(RetrievalFilter("metadata.value", (expected,)),)
        )
    ] == ["doc"]
