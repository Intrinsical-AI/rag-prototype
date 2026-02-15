from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_DIST_NAME = "rag-prototype"

try:
    __version__ = version(_DIST_NAME)
except PackageNotFoundError:  # pragma: no cover
    # Allows running directly from source without an installed dist.
    __version__ = "0.0.0"

__all__ = ["__version__"]
