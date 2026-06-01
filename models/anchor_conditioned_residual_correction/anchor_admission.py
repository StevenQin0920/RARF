from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorAdmissionGate(nn.Module):
    """Node-wise admission of recent high-frequency evidence into the anchor generator."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        r_min: float,
        r_max: float,
        r_init: float,
    ) -> None:
        super().__init__()
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("AnchorAdmissionGate requires positive dimensions.")
        if not (0.0 <= self.r_min < self.r_max):
            raise ValueError("anchor_admission_gate_r_min/r_max must satisfy 0 <= min < max.")
        if not (self.r_min <= r_init <= self.r_max):
            raise ValueError("anchor_admission_gate_r_init must lie within [r_min, r_max].")
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.reset_parameters(r_init)

    def reset_parameters(self, r_init: float) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        ratio = (float(r_init) - self.r_min) / max(self.r_max - self.r_min, 1e-8)
        ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
        nn.init.constant_(self.fc2.bias, torch.logit(torch.tensor(ratio)).item())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(self.norm(features)))
        gate = torch.sigmoid(self.fc2(hidden)).squeeze(-1)
        return self.r_min + (self.r_max - self.r_min) * gate
