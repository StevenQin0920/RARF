from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_adjacency_tensor(value, expected_nodes: int, name: str) -> torch.Tensor:
    if value is None:
        return torch.eye(expected_nodes, dtype=torch.float32)
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() != 2 or tuple(tensor.shape) != (expected_nodes, expected_nodes):
        raise ValueError(f"{name} must have shape {(expected_nodes, expected_nodes)}, got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


class SpatialBiasSelfAttention(nn.Module):
    """Self-attention block used by the Spatial-Bias Correction Branch."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        input_dim = int(input_dim)
        if input_dim <= 0:
            raise ValueError("Spatial-Bias Correction Branch attention input_dim must be positive.")
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(key.shape[-1]))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, value)


class SpatialBiasCorrectionPattern(nn.Module):
    """Context-level spatial residual correction around a frozen anchor bank."""

    LOWRANK_RANK = 32
    LOWRANK_LAMBDA_MAX = 0.1

    def __init__(
        self,
        *,
        context_slot_count: int,
        num_nodes: int,
        adj,
        anchor_bank: torch.Tensor,
        name: str,
    ) -> None:
        super().__init__()
        self.context_slot_count = int(context_slot_count)
        self.num_nodes = int(num_nodes)
        if self.context_slot_count <= 0 or self.num_nodes <= 0:
            raise ValueError(f"{name} requires positive context_slot_count and num_nodes.")
        anchor_bank = torch.as_tensor(anchor_bank, dtype=torch.float32)
        if tuple(anchor_bank.shape) != (self.context_slot_count, self.num_nodes):
            raise ValueError(
                f"{name} anchor_bank must have shape {(self.context_slot_count, self.num_nodes)}, "
                f"got {tuple(anchor_bank.shape)}."
            )
        if not torch.isfinite(anchor_bank).all():
            raise ValueError(f"{name} anchor_bank contains non-finite values.")
        self.register_buffer("anchor_bank", anchor_bank.detach().clone())
        self.residual_code = nn.Parameter(torch.zeros_like(anchor_bank))
        self.register_buffer("adj", _as_adjacency_tensor(adj, self.num_nodes, f"{name} adjacency"))
        self.linear1 = nn.Linear(self.num_nodes, self.num_nodes)
        self.linear2 = nn.Linear(2 * self.num_nodes, self.num_nodes)
        self.attention_t = SpatialBiasSelfAttention(self.num_nodes)
        self.attention_s = SpatialBiasSelfAttention(self.context_slot_count)
        self.reset_parameters()
        self._init_lowrank_adapter()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.residual_code)

    def _init_lowrank_adapter(self) -> None:
        rng_state = torch.random.get_rng_state()
        self.regime_factor = nn.Parameter(torch.empty(self.context_slot_count, self.LOWRANK_RANK))
        self.node_factor = nn.Parameter(torch.empty(self.num_nodes, self.LOWRANK_RANK))
        self.lowrank_lambda_raw = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.regime_factor)
        nn.init.xavier_uniform_(self.node_factor)
        nn.init.zeros_(self.lowrank_lambda_raw)
        torch.random.set_rng_state(rng_state)

    def lowrank_lambda(self) -> torch.Tensor:
        return self.LOWRANK_LAMBDA_MAX * torch.tanh(self.lowrank_lambda_raw)

    def lowrank_component(self) -> torch.Tensor:
        raw = torch.matmul(self.regime_factor, self.node_factor.transpose(0, 1))
        raw = raw / math.sqrt(float(self.LOWRANK_RANK))
        return torch.tanh(raw)

    def lowrank_adapter(self) -> torch.Tensor:
        return self.lowrank_lambda() * self.lowrank_component()

    def branch_input(self) -> torch.Tensor:
        return self.anchor_bank + self.residual_code + self.lowrank_adapter()

    def transformed_reference(self, field: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=self.anchor_bank.device.type, enabled=False):
            base = field.float()
            adj = self.adj.to(device=base.device, dtype=torch.float32)
            data = torch.einsum("ln,nv->lv", base, adj)
            data = F.relu(self.linear1(data))
            data_t = self.attention_t(data)
            data_s = self.attention_s(data.transpose(0, 1)).transpose(0, 1)
            return self.linear2(torch.cat([data_t, data_s], dim=-1))

    def corrected_reference_bank(self) -> torch.Tensor:
        return self.transformed_reference(self.branch_input())

    def correction_bank(self) -> torch.Tensor:
        return self.corrected_reference_bank() - self.anchor_bank

    def lookup_corrected_reference(self, index_ids: torch.Tensor) -> torch.Tensor:
        index_ids = index_ids.long().clamp(0, self.context_slot_count - 1)
        return self.corrected_reference_bank()[index_ids]

    def lookup_correction(self, index_ids: torch.Tensor) -> torch.Tensor:
        index_ids = index_ids.long().clamp(0, self.context_slot_count - 1)
        return self.correction_bank()[index_ids]


class SpatialBiasCorrectionBranch(nn.Module):
    """Spatial-Bias Correction Branch for context-level correction around A0.

    The branch keeps the train-only anchor A0 frozen and learns:

        R_spatial = T_phi(A0 + E_table + lambda * LowRank(node, regime)) - A0

    The model uses A0 + R_spatial as the behavior-equivalent reference while
    exposing R_spatial as part of Anchor-Conditioned Residual Correction.
    """

    def __init__(
        self,
        *,
        daily_bank: torch.Tensor,
        weekly_bank: torch.Tensor,
        tod_size: int,
        dow_size: int,
        adj,
    ) -> None:
        super().__init__()
        daily = torch.as_tensor(daily_bank, dtype=torch.float32)
        weekly = torch.as_tensor(weekly_bank, dtype=torch.float32)
        self.tod_size = int(tod_size)
        self.dow_size = int(dow_size)
        self.weekly_size = self.tod_size * self.dow_size
        self.num_nodes = int(daily.shape[1]) if daily.dim() == 2 else 0
        if tuple(daily.shape) != (self.tod_size, self.num_nodes):
            raise ValueError(f"daily_bank must have shape {(self.tod_size, self.num_nodes)}, got {tuple(daily.shape)}.")
        if tuple(weekly.shape) != (self.weekly_size, self.num_nodes):
            raise ValueError(f"weekly_bank must have shape {(self.weekly_size, self.num_nodes)}, got {tuple(weekly.shape)}.")
        self.daily_branch = SpatialBiasCorrectionPattern(
            context_slot_count=self.tod_size,
            num_nodes=self.num_nodes,
            adj=adj,
            anchor_bank=daily,
            name="daily spatial-bias correction branch",
        )
        self.weekly_branch = SpatialBiasCorrectionPattern(
            context_slot_count=self.weekly_size,
            num_nodes=self.num_nodes,
            adj=adj,
            anchor_bank=weekly,
            name="weekly spatial-bias correction branch",
        )

    def daily_reference_bank(self) -> torch.Tensor:
        return self.daily_branch.corrected_reference_bank()

    def weekly_reference_bank(self) -> torch.Tensor:
        return self.weekly_branch.corrected_reference_bank()

    def daily_correction_bank(self) -> torch.Tensor:
        return self.daily_branch.correction_bank()

    def weekly_correction_bank(self) -> torch.Tensor:
        return self.weekly_branch.correction_bank()

    def lookup_corrected_components(self, tod_ids: torch.Tensor, week_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tod_ids = tod_ids.long().clamp(0, self.tod_size - 1)
        week_ids = week_ids.long().clamp(0, self.weekly_size - 1)
        return self.daily_branch.lookup_corrected_reference(tod_ids), self.weekly_branch.lookup_corrected_reference(week_ids)

    def lookup_correction_components(self, tod_ids: torch.Tensor, week_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tod_ids = tod_ids.long().clamp(0, self.tod_size - 1)
        week_ids = week_ids.long().clamp(0, self.weekly_size - 1)
        return self.daily_branch.lookup_correction(tod_ids), self.weekly_branch.lookup_correction(week_ids)
