"""Repo entrypoint: bootstrap sample data into the configured SQLite (and FAISS if enabled).

This wrapper exists for convenience when working from the repository.
The installable entrypoint remains `rag-bootstrap`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a repo checkout without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir():
    sys.path.insert(0, str(_SRC_DIR))


if __name__ == "__main__":
    from local_rag_backend.scripts.bootstrap import main

    main()
