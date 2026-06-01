from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMultiSupportGraphBlock(nn.Module):
    """Graph block over fixed physical supports."""

    VALID_SUPPORTS = {"identity", "forward", "backward"}

    def __init__(
        self,
        hidden_dim: int,
        support_names: Tuple[str, ...],
        dropout: float = 0.0,
        use_gate: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        unknown = sorted(set(support_names) - self.VALID_SUPPORTS)
        if unknown:
            raise ValueError(f"Unknown graph supports {unknown}. Valid supports: {sorted(self.VALID_SUPPORTS)}")
        if not support_names:
            raise ValueError("ResidualMultiSupportGraphBlock requires at least one support.")
        self.support_names = tuple(support_names)
        self.use_gate = bool(use_gate)
        self.message_projection = nn.Linear(hidden_dim * len(self.support_names), hidden_dim)
        self.gate_projection = nn.Linear(2 * hidden_dim, hidden_dim) if self.use_gate else None
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    @staticmethod
    def _propagate(x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        if graph.dim() == 2:
            return torch.einsum("ij,bljd->blid", graph, x)
        if graph.dim() == 3:
            return torch.einsum("bij,bljd->blid", graph, x)
        raise ValueError(f"Expected graph with 2 or 3 dims, got {graph.dim()}")

    def _support_feature(self, name: str, x: torch.Tensor, graph: torch.Tensor, graph_t: torch.Tensor) -> torch.Tensor:
        if name == "identity":
            return x
        if name == "forward":
            return self._propagate(x, graph)
        if name == "backward":
            return self._propagate(x, graph_t)
        raise RuntimeError(f"Unhandled graph support `{name}`")

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        graph_t = graph.transpose(-1, -2)
        support_features = [self._support_feature(name, x, graph, graph_t) for name in self.support_names]
        message = torch.cat(support_features, dim=-1)
        message = F.gelu(self.message_projection(message))
        message = self.dropout(message)
        if self.gate_projection is not None:
            gate = torch.sigmoid(self.gate_projection(torch.cat([x, message], dim=-1)))
            message = gate * message
        return self.norm(x + message)
