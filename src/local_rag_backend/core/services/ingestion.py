from typing import Callable, Sequence
from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.core.ports import LoaderPort
from local_rag_backend.utils import preprocess_text


def default_preprocess(text: str, metadata: dict | None = None) -> str:
    return preprocess_text(text)


def default_chunker(max_chars: int = 1200, overlap: int = 200):
    def _chunk(t: str) -> list[str]:
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


def default_formatter(chunk: str, metadata: dict | None = None) -> str:
    if not metadata:
        return chunk
    header = "\n".join(f"{k.title()}: {v}" for k, v in metadata.items() if v is not None)
    return f"{header}\n\n{chunk}" if header else chunk


class IngestionPipeline:
    def __init__(
        self,
        loader: LoaderPort,
        etl: ETLService,
        preprocess: Callable[[str, dict | None], str] = default_preprocess,
        chunk: Callable[[str], list[str]] = default_chunker(),
        format_chunk: Callable[[str, dict | None], str] = default_formatter,
        batch_size: int = 128,
    ):
        self.loader = loader
        self.etl = etl
        self.preprocess = preprocess
        self.chunk = chunk
        self.format_chunk = format_chunk
        self.batch_size = batch_size

    def run(self) -> list[int]:
        buf, ids = [], []
        for item in self.loader.load():
            clean = self.preprocess(item.text, item.metadata or None)
            for c in self.chunk(clean):
                buf.append(self.format_chunk(c, item.metadata or None))
                if len(buf) >= self.batch_size:
                    ids += list(self.etl.ingest(buf))
                    buf.clear()
        if buf:
            ids += list(self.etl.ingest(buf))
        return ids
