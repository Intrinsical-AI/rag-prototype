# src/infrastructure/ingestion/loaders/csv_loader.py
"""
CSV loader for basic document ingestion.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.core.ports import LoaderPort
from local_rag_backend.infrastructure.ingestion.loaders.lineage import loader_lineage

if TYPE_CHECKING:
    from collections.abc import Iterable


class CSVLoader(LoaderPort):
    def __init__(
        self, path: str | Path, delimiter: str | None = None, has_header: bool = True
    ) -> None:
        self.path = Path(path)
        self.delimiter = delimiter
        self.has_header = has_header

    def load(self) -> Iterable[LoadedItem]:
        with self.path.open(encoding="utf-8", newline="") as fh:
            delimiter = self.delimiter
            if delimiter is None:
                sample = fh.read(4096)
                fh.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    delimiter = str(dialect.delimiter)
                except Exception:
                    delimiter = ";"

            reader = csv.reader(fh, delimiter=delimiter)
            if self.has_header:
                next(reader, None)
            row_index = 0
            for row in reader:
                if not row:
                    continue
                row_index += 1
                if len(row) >= 2:
                    title, body = row[0].strip(), row[1].strip()
                    text = f"{title}\n\n{body}"
                    metadata = {"title": title, "row_index": row_index}
                else:
                    text = row[0].strip()
                    metadata = {"row_index": row_index}
                yield LoadedItem(
                    text=text,
                    lineage=loader_lineage(
                        source_uri=str(self.path.resolve()),
                        loader_name="CSVLoader",
                        record_locator=f"row:{row_index}",
                    ),
                    metadata=metadata,
                )
