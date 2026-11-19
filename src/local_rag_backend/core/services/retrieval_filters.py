"""Transport-neutral validation for structured retrieval filter values."""

from __future__ import annotations

from local_rag_backend.core.domain.retrieval import RetrievalFilter


def parse_filters(value: object) -> tuple[RetrievalFilter, ...]:
    """Decode the shared JSON filter input without transport or storage dependencies."""
    if not isinstance(value, list):
        raise ValueError("filters must be a list")
    result: list[RetrievalFilter] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"field", "values"}:
            raise ValueError("Each filter must contain field and values")
        if not isinstance(item["field"], str):
            raise ValueError("Filter field must be a string")
        values = validate_filter_values(item["values"])
        result.append(RetrievalFilter(field=item["field"], values=tuple(values)))
    return tuple(result)


def validate_filter_values(value: object) -> list[str]:
    """Accept a nonempty JSON array of nonblank strings and trim each value."""
    if not isinstance(value, list) or not value:
        raise ValueError("filter values must be a nonempty array of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("filter values must contain only nonblank strings")
    return [item.strip() for item in value]
