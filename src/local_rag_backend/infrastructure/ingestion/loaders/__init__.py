# Loader implementations

import hashlib
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.core.ports import LoaderPort

__all__ = ["DirectoryLoader", "UniqueLoader", "WebPageLoader"]


class DirectoryLoader(LoaderPort):
    def __init__(
        self,
        root: str | Path,
        patterns: Iterable[str] | None = None,
        files: Iterable[str | Path] | None = None,
    ):
        self.root = Path(root)
        self.patterns = list(patterns) if patterns else ["**/*.md", "**/*.txt"]
        self._files = [Path(p) for p in files] if files is not None else None

    def load(self) -> Iterable[LoadedItem]:
        seen: set[Path] = set()
        if self._files is not None:
            candidates: Sequence[Path] = [p for p in self._files if Path(p).is_file()]
        else:
            candidates = []
            for pattern in self.patterns:
                candidates.extend(self.root.rglob(pattern))
        for path in candidates:
            if not path.is_file():
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                print("Failed to read file: %s", path)
                continue
            try:
                rel = str(path.relative_to(self.root))
            except Exception:
                rel = str(path)
            yield LoadedItem(text=text, metadata={"path": rel, "source": "fs"})


class WebPageLoader(LoaderPort):
    def __init__(self, urls: list[str], timeout: int = 20, workers: int = 1):
        self.urls = urls
        self.timeout = timeout
        self.workers = max(1, int(workers))

    def _fetch_one(self, url: str) -> tuple[str, str | None]:
        import trafilatura  # lazy import to keep optional

        try:
            downloaded = trafilatura.fetch_url(url, timeout=self.timeout)
            text = trafilatura.extract(downloaded) if downloaded else None
            return url, text
        except Exception:
            return url, None

    def load(self) -> Iterable[LoadedItem]:
        try:
            import trafilatura  # noqa: F401  # ensure importable
        except Exception as e:
            raise RuntimeError(
                "trafilatura is not installed. Install the 'loaders' extra: pip install intrinsical-rag-prototype[loaders]"
            ) from e

        if self.workers > 1 and len(self.urls) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                results = list(ex.map(self._fetch_one, self.urls))
            # preserve original order
            for url, text in results:
                if text:
                    yield LoadedItem(text=text, metadata={"url": url, "source": "web"})
        else:
            for url in self.urls:
                _, text = self._fetch_one(url)
                if text:
                    yield LoadedItem(text=text, metadata={"url": url, "source": "web"})


class UniqueLoader(LoaderPort):
    """Wrapper loader that removes duplicate texts within a single run.

    Deduplication is performed via SHA1 of the `text` field. This is a
    best-effort, in-memory dedup that avoids re-ingesting identical chunks
    during one CLI run. It does not persist across runs.
    """

    def __init__(self, inner: LoaderPort):
        self.inner = inner

    def load(self) -> Iterable[LoadedItem]:
        seen: set[str] = set()
        for item in self.inner.load():
            digest = hashlib.sha1(
                item.text.encode("utf-8", errors="ignore"), usedforsecurity=False
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            yield item
