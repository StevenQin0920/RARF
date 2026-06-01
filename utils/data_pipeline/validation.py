from __future__ import annotations

from pathlib import Path

import numpy as np


def require_existing_file(path: Path | str, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    return resolved


def require_array_rank(array: np.ndarray, rank: int, label: str) -> None:
    if array.ndim != int(rank):
        raise ValueError(f"{label} must have rank {rank}, got shape {array.shape}.")


def require_nonempty_axis(array: np.ndarray, axis: int, label: str) -> None:
    if array.shape[int(axis)] <= 0:
        raise ValueError(f"{label} axis {axis} must be non-empty, got shape {array.shape}.")
