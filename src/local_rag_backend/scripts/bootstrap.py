# scripts/bootstrap.py
"""Bootstrap sample data into SQLite/FAISS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from local_rag_backend.scripts.sample_data_ingestion import run_sample_data_ingestion
from local_rag_backend.settings import settings as default_settings

if TYPE_CHECKING:
    from pathlib import Path


def main(csv_path: str | Path | None = None, **kwargs: Any) -> None:
    settings_obj = kwargs.get("settings", default_settings)
    run_sample_data_ingestion(csv_path=csv_path, settings_obj=settings_obj)


if __name__ == "__main__":
    main()
