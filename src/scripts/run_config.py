"""Shared runtime path and hyperparameter helpers.

Paths are repository-relative by default and can be overridden with the
documented ``ECG_*`` environment variables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_stdio()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _prefer_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def default_data_dir() -> Path:
    env_path = os.environ.get("ECG_DATA_DIR")
    if env_path:
        return Path(env_path)
    return _prefer_existing(
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "datasets" / "processed",
    )


def default_raw_data_dir() -> Path:
    env_path = os.environ.get("ECG_RAW_DATA_DIR")
    if env_path:
        return Path(env_path)
    return _prefer_existing(
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "datasets" / "raw",
    )


def default_output_dir() -> Path:
    env_path = os.environ.get("ECG_OUTPUT_DIR")
    if env_path:
        return Path(env_path)
    return PROJECT_ROOT / "outputs"


def checkpoint_candidates(extra: Iterable[Path | str] = ()) -> List[Path]:
    candidates: List[Path] = [Path(p) for p in extra]
    env_path = os.environ.get("ECG_CKPT_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            PROJECT_ROOT / "checkpoints" / "1_lead_ECGFounder.pth",
            PROJECT_ROOT / "assets" / "1_lead_ECGFounder.pth",
        ]
    )
    return candidates


def default_checkpoint_path() -> Path:
    candidates = checkpoint_candidates()
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
