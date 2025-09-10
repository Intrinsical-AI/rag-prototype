import csv
from pathlib import Path
from typing import Iterable
from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.core.ports import LoaderPort


class CSVLoader(LoaderPort):
    def __init__(self, path: str | Path, delimiter: str = ";", has_header: bool = True):
        self.path = Path(path)
        self.delimiter = delimiter
        self.has_header = has_header

    def load(self) -> Iterable[LoadedItem]:
        with self.path.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter=self.delimiter)
            if self.has_header:
                next(reader, None)
            for row in reader:
                if not row:
                    continue
                if len(row) >= 2:
                    title, body = row[0].strip(), row[1].strip()
                    yield LoadedItem(text=f"{title}\n\n{body}", metadata={"title": title})
                else:
                    yield LoadedItem(text=row[0].strip(), metadata=None)
