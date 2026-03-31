"""Minimal logging helpers for reproducible experiments."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import ensure_dir


def build_logger(log_dir: str | Path, name: str = "lmca_tic") -> logging.Logger:
    log_path = ensure_dir(log_dir) / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def capture_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _safe_git_hash(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_git_hash() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return output.decode("utf-8").strip()
    except Exception:
        return "not-a-git-repository"
