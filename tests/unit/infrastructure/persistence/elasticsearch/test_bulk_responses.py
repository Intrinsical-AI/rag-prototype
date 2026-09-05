"""Bulk exceptions are scoped to action ordinals, never shared document IDs."""

import httpx
import pytest

from local_rag_backend.infrastructure.persistence.elasticsearch.client import (
    ElasticBackendError,
    ElasticClient,
)


def _client(response):
    return ElasticClient(
        client=httpx.Client(
            base_url="http://example.test",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
        )
    )


def _missing(error_type="document_missing_exception", status=404):
    return {"update": {"_id": "same", "status": status, "error": {"type": error_type}}}


def _ops():
    return [
        {"delete": {"_id": "other", "_index": "docs"}},
        {"index": {"_id": "new", "_index": "docs"}},
        {"content": "data line", "update": "also just data"},
        {"update": {"_id": "same", "_index": "docs"}},
        {"script": {"source": "ctx._source.remove('embedding')"}},
        {"update": {"_id": "same", "_index": "docs"}},
        {"doc": {"embedding": [1.0]}},
    ]


def _response(last=None):
    return {
        "errors": True,
        "items": [
            {"delete": {"_id": "other", "status": 200}},
            {"index": {"_id": "new", "status": 201}},
            _missing(),
            last or {"update": {"_id": "same", "status": 200}},
        ],
    }


def test_only_marked_action_ordinal_tolerates_missing_document():
    result = _client(_response()).bulk(_ops(), missing_vector_delete_ordinals={2})
    assert result["errors"] is True
    with pytest.raises(ElasticBackendError):
        _client(_response()).bulk(_ops(), missing_vector_delete_ordinals={3})
    with pytest.raises(ElasticBackendError):
        _client(_response(_missing())).bulk(_ops(), missing_vector_delete_ordinals={2})


@pytest.mark.parametrize(
    "error_type,status",
    [
        ("index_not_found_exception", 404),
        ("document_missing_exception", 403),
        ("document_missing_exception", 409),
        ("document_missing_exception", 429),
        ("unknown_exception", 404),
    ],
)
def test_other_vector_removal_errors_remain_strict(error_type, status):
    response = {"errors": True, "items": [_missing(error_type, status)]}
    with pytest.raises(ElasticBackendError, match=f"status {status}"):
        _client(response).bulk(_ops()[3:5], missing_vector_delete_ordinals={0})


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"errors": False},
        {"errors": False, "items": []},
        {"errors": True, "items": []},
        {"errors": True, "items": None},
        {"errors": False, "items": [{}]},
        {"errors": False, "items": [{"update": {"_id": "same"}}]},
        {"errors": False, "items": [{"update": {"_id": "same", "status": "invalid"}}]},
        {"errors": False, "items": [{"delete": {"_id": "same", "status": 200}}]},
        {"errors": False, "items": [{"update": {"_id": "different", "status": 200}}]},
        {"errors": True, "items": [{"update": {"_id": "same", "status": 200}}]},
        {"items": [_missing()]},
    ],
)
def test_incomplete_or_unexplained_bulk_responses_fail(response):
    with pytest.raises(ElasticBackendError):
        _client(response).bulk(_ops()[3:5], missing_vector_delete_ordinals={0})
