from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_path = str(src_dir)
    if src_dir.exists() and src_path not in sys.path:
        sys.path.insert(0, src_path)
