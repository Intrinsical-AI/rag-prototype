from collections.abc import Callable
from typing import Any

from local_rag_backend.core.ports import LoaderPort
from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.utils import preprocess_text


def default_preprocess(text: str, metadata: dict[str, Any] | None = None) -> str:  # noqa: ARG001
    return preprocess_text(text)


def default_chunker(
    max_chars: int = 1000, overlap: int = 100
) -> Callable[[str, dict[str, Any] | None], list[str]]:
    if overlap >= max_chars:
        # Clamp overlap to prevent infinite loop
        overlap = max_chars - 1

    def _chunk(t: str, metadata: dict[str, Any] | None = None) -> list[str]:  # noqa: ARG001
        if len(t) <= max_chars:
            return [t]
        out, i = [], 0
        while i < len(t):
            j = min(i + max_chars, len(t))
            out.append(t[i:j])
            if j == len(t):
                break
            i = max(0, j - overlap)
        return out

    return _chunk


def default_formatter(text: str, metadata: dict[str, Any] | None = None) -> str:
    if not metadata:
        return text
    header = "\n".join(f"{k.title()}: {v}" for k, v in metadata.items() if v is not None)
    return f"{header}\n\n{text}" if header else text


class IngestionPipeline:
    def __init__(
        self,
        loader: LoaderPort,
        etl: ETLService,
        preprocess: Callable[[str, dict[str, Any] | None], str] | None = None,
        chunk: Callable[[str, dict[str, Any] | None], list[str]] | None = None,
        format_chunk: Callable[[str, dict[str, Any] | None], str] | None = None,
        batch_size: int = 128,
    ):
        self.loader = loader
        self.etl = etl
        self.preprocess = preprocess or default_preprocess
        self.chunk = chunk or default_chunker()
        self.format_chunk = format_chunk or default_formatter
        self.batch_size = batch_size

    def run(self) -> list[int]:
        buf, ids = [], []
        for item in self.loader.load():
            clean = self.preprocess(item.text, dict(item.metadata) if item.metadata else None)
            for c in self.chunk(clean, dict(item.metadata) if item.metadata else None):
                buf.append(self.format_chunk(c, dict(item.metadata) if item.metadata else None))
                if len(buf) >= self.batch_size:
                    ids += list(self.etl.ingest(buf))
                    buf.clear()
        if buf:
            ids += list(self.etl.ingest(buf))
        return ids
