from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..anchor_conditioned_residual_correction.spatial_bias_correction_branch import SpatialBiasCorrectionBranch


def _load_anchor_bank(path: Optional[str], value, expected_shape: Tuple[int, int], name: str) -> torch.Tensor:
    if value is None:
        if path is None:
            raise ValueError(f"{name} requires either an in-memory array or a .npy path.")
        array = np.load(Path(path))
    else:
        array = value
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


class FrozenRegimeAnchor(nn.Module):
    """Frozen train-only anchor A0 plus the Spatial-Bias Correction Branch."""

    def __init__(
        self,
        *,
        num_nodes: int,
        tod_vocab_size: int,
        dow_vocab_size: int,
        horizon: int,
        daily_init_path: Optional[str] = None,
        weekly_init_path: Optional[str] = None,
        daily_init=None,
        weekly_init=None,
        daily_weight: float = 1.0,
        weekly_weight: float = 1.0,
        spatial_bias_graph_adj=None,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.tod_vocab_size = int(tod_vocab_size)
        self.dow_vocab_size = int(dow_vocab_size)
        self.weekly_vocab_size = self.tod_vocab_size * self.dow_vocab_size
        self.horizon = int(horizon)
        self.daily_weight = float(daily_weight)
        self.weekly_weight = float(weekly_weight)

        if self.num_nodes <= 0 or self.tod_vocab_size <= 0 or self.dow_vocab_size <= 0 or self.horizon <= 0:
            raise ValueError("Frozen Regime Anchor requires positive node/time/horizon sizes.")
        if self.daily_weight == 0.0 and self.weekly_weight == 0.0:
            raise ValueError("At least one of daily_weight or weekly_weight must be non-zero.")

        daily_anchor_bank = (
            torch.zeros(self.tod_vocab_size, self.num_nodes, dtype=torch.float32)
            if self.daily_weight == 0.0
            else _load_anchor_bank(
                daily_init_path,
                daily_init,
                (self.tod_vocab_size, self.num_nodes),
                "daily frozen anchor",
            )
        )
        weekly_anchor_bank = (
            torch.zeros(self.weekly_vocab_size, self.num_nodes, dtype=torch.float32)
            if self.weekly_weight == 0.0
            else _load_anchor_bank(
                weekly_init_path,
                weekly_init,
                (self.weekly_vocab_size, self.num_nodes),
                "weekly frozen anchor residual",
            )
        )
        self.register_buffer("daily_anchor_bank", daily_anchor_bank)
        self.register_buffer("weekly_anchor_bank", weekly_anchor_bank)

        self.spatial_bias_branch = SpatialBiasCorrectionBranch(
            daily_bank=daily_anchor_bank,
            weekly_bank=weekly_anchor_bank,
            tod_size=self.tod_vocab_size,
            dow_size=self.dow_vocab_size,
            adj=spatial_bias_graph_adj,
        )

    def _regime_ids(self, tod_ids: torch.Tensor, dow_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if tod_ids.dim() == 3:
            tod_ids = tod_ids[..., 0]
        if dow_ids.dim() == 3:
            dow_ids = dow_ids[..., 0]
        if tod_ids.shape != dow_ids.shape:
            raise ValueError(f"tod_ids and dow_ids must match, got {tuple(tod_ids.shape)} and {tuple(dow_ids.shape)}.")
        tod_ids = tod_ids.long().clamp(0, self.tod_vocab_size - 1)
        dow_ids = dow_ids.long().clamp(0, self.dow_vocab_size - 1)
        week_ids = (dow_ids * self.tod_vocab_size + tod_ids).clamp(0, self.weekly_vocab_size - 1)
        return tod_ids, week_ids

    def lookup_a0_components(self, tod_ids: torch.Tensor, dow_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tod_ids, week_ids = self._regime_ids(tod_ids, dow_ids)
        daily = self.daily_weight * self.daily_anchor_bank[tod_ids]
        weekly = self.weekly_weight * self.weekly_anchor_bank[week_ids]
        return daily, weekly

    def lookup_spatial_bias_components(
        self,
        tod_ids: torch.Tensor,
        dow_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tod_ids, week_ids = self._regime_ids(tod_ids, dow_ids)
        daily, weekly = self.spatial_bias_branch.lookup_correction_components(tod_ids, week_ids)
        return self.daily_weight * daily, self.weekly_weight * weekly

    def lookup_corrected_reference_components(
        self,
        tod_ids: torch.Tensor,
        dow_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tod_ids, week_ids = self._regime_ids(tod_ids, dow_ids)
        daily, weekly = self.spatial_bias_branch.lookup_corrected_components(tod_ids, week_ids)
        return self.daily_weight * daily, self.weekly_weight * weekly
