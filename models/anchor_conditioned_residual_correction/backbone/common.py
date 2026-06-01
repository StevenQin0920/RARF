from __future__ import annotations

from typing import Union

import torch

TensorLike = Union[torch.Tensor, list, tuple]

def _as_float_tensor(value: TensorLike) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("prior adjacency list is empty")
        value = value[0]
    return torch.as_tensor(value, dtype=torch.float32)

def row_normalize(adj: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Row-normalize a static or batched adjacency matrix."""
    degree = adj.sum(dim=-1, keepdim=True).clamp_min(eps)
    return adj / degree


