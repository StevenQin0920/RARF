from __future__ import annotations

import torch
import torch.nn as nn


class FutureContextEncoder(nn.Module):
    """Encode known future TOD/DOW/weekend/holiday context."""

    def __init__(
        self,
        tod_vocab_size: int,
        dow_vocab_size: int,
        weekend_vocab_size: int,
        holiday_vocab_size: int,
        tod_emb_dim: int,
        dow_emb_dim: int,
        weekend_emb_dim: int,
        holiday_emb_dim: int,
        output_dim: int,
        use_weekend: bool = True,
        use_holiday: bool = True,
    ):
        super().__init__()
        self.use_weekend = bool(use_weekend)
        self.use_holiday = bool(use_holiday)
        self.tod_embedding = nn.Embedding(tod_vocab_size, tod_emb_dim)
        self.dow_embedding = nn.Embedding(dow_vocab_size, dow_emb_dim)
        self.weekend_embedding = nn.Embedding(weekend_vocab_size, weekend_emb_dim) if self.use_weekend else None
        self.holiday_embedding = nn.Embedding(holiday_vocab_size, holiday_emb_dim) if self.use_holiday else None
        input_dim = tod_emb_dim + dow_emb_dim
        if self.use_weekend:
            input_dim += weekend_emb_dim
        if self.use_holiday:
            input_dim += holiday_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        tod_t: torch.Tensor,
        dow_t: torch.Tensor,
        weekend_t: torch.Tensor,
        holiday_t: torch.Tensor,
    ) -> torch.Tensor:
        parts = [self.tod_embedding(tod_t), self.dow_embedding(dow_t)]
        if self.weekend_embedding is not None:
            parts.append(self.weekend_embedding(weekend_t))
        if self.holiday_embedding is not None:
            parts.append(self.holiday_embedding(holiday_t))
        hidden = torch.cat(parts, dim=-1)
        return self.mlp(hidden)
