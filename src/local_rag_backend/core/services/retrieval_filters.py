"""Transport-neutral validation for structured retrieval filter values."""

from __future__ import annotations


def validate_filter_values(value: object) -> list[str]:
    """Accept a nonempty JSON array of nonblank strings and trim each value."""
    if not isinstance(value, list) or not value:
        raise ValueError("filter values must be a nonempty array of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("filter values must contain only nonblank strings")
    return [item.strip() for item in value]
