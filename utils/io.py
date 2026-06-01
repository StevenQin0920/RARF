from __future__ import annotations

import pickle
import warnings
from typing import Optional

import numpy as np
import torch

from .config import resolve_path

def load_adj_matrix(path: str) -> np.ndarray:
    resolved = resolve_path(path)
    if resolved is None:
        raise ValueError("adjacency path is not configured")
    visible_deprecation = getattr(np, "VisibleDeprecationWarning", None)
    if visible_deprecation is None:
        visible_deprecation = getattr(getattr(np, "exceptions", object), "VisibleDeprecationWarning", None)
    with open(resolved, "rb") as f:
        with warnings.catch_warnings():
            if visible_deprecation is not None:
                warnings.filterwarnings(
                    "ignore",
                    message=r"dtype\(\): align should be passed as Python or NumPy boolean but got `align=0`.*",
                    category=visible_deprecation,
                )
            try:
                data = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                data = pickle.load(f, encoding="latin1")
    if isinstance(data, (list, tuple)) and len(data) >= 3:
        return np.asarray(data[2], dtype=np.float32)
    return np.asarray(data, dtype=np.float32)

def iter_batches(loader, device: torch.device, max_batches: Optional[int] = None):
    for batch_idx, (x, y) in enumerate(loader.get_iterator()):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch_idx, torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device)


