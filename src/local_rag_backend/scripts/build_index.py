# scripts/build_index.py
"""Build index command alias backed by shared sample-data ingestion implementation."""

from __future__ import annotations

from local_rag_backend.scripts.sample_data_ingestion import run_sample_data_ingestion
from local_rag_backend.settings import settings


def main() -> None:
    run_sample_data_ingestion(
        settings_obj=settings,
        schema_error_message="Unable to ensure SQLite schema before build-index.",
    )


if __name__ == "__main__":
    main()
