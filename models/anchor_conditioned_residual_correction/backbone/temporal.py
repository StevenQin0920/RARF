from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import ResidualMultiSupportGraphBlock

class CausalTemporalConv(nn.Module):
    """A small causal temporal convolution block for traffic state encoding."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x + residual

class GatedCausalTemporalConv(nn.Module):
    """Gated causal temporal block for stronger traffic state encoding."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.gate_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels) if use_layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        padded = F.pad(x, (self.left_padding, 0))
        hidden = torch.tanh(self.filter_conv(padded)) * torch.sigmoid(self.gate_conv(padded))
        hidden = residual + self.dropout(hidden)
        hidden = hidden.transpose(1, 2)
        hidden = self.norm(hidden)
        return hidden.transpose(1, 2)

class TemporalConvEncoder(nn.Module):
    """Temporal encoder used only to estimate traffic state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
        encoder_type: str = "causal_tcn",
        dilations: Optional[Tuple[int, ...]] = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if encoder_type not in {"causal_tcn", "gated_tcn"}:
            raise ValueError("temporal_encoder_type must be one of: causal_tcn, gated_tcn")
        self.encoder_type = encoder_type
        projection_dim = input_dim + 2 * time_emb_dim
        if dilations is None:
            dilations = tuple(2**layer_idx for layer_idx in range(num_layers))
        else:
            dilations = tuple(int(dilation) for dilation in dilations)
            if len(dilations) != num_layers:
                raise ValueError(
                    f"temporal_dilations length {len(dilations)} must match temporal_layers {num_layers}."
                )
        block_cls = GatedCausalTemporalConv if self.encoder_type == "gated_tcn" else CausalTemporalConv
        self.input_projection = nn.Linear(projection_dim, hidden_dim)
        self.layers = self._build_layers(block_cls, hidden_dim, kernel_size, dropout, dilations, use_layer_norm)

    @staticmethod
    def _build_layers(
        block_cls,
        hidden_dim: int,
        kernel_size: int,
        dropout: float,
        dilations: Tuple[int, ...],
        use_layer_norm: bool,
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                block_cls(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    **({"use_layer_norm": use_layer_norm} if block_cls is GatedCausalTemporalConv else {}),
                )
                for dilation in dilations
            ]
        )

    def forward(
        self,
        traffic: torch.Tensor,
        time_in_day: torch.Tensor,
        day_in_week: torch.Tensor,
    ) -> torch.Tensor:
        # traffic/time tensors: [B, L, N, D]
        inputs = torch.cat([traffic, time_in_day, day_in_week], dim=-1)
        hidden = self.input_projection(inputs)

        batch_size, seq_len, num_nodes, hidden_dim = hidden.shape
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, hidden_dim, seq_len)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = hidden.reshape(batch_size, num_nodes, hidden_dim, seq_len).permute(0, 3, 1, 2)
        return hidden

class AlternatingSpatioTemporalBlock(nn.Module):
    """One anchor-support temporal/spatial refinement block for H_t encoding."""

    def __init__(
        self,
        hidden_dim: int,
        temporal_kernel_size: int,
        temporal_dilation: int,
        temporal_dropout: float,
        spatial_supports: Tuple[str, ...],
        spatial_dropout: float,
        spatial_gate: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.temporal = GatedCausalTemporalConv(
            channels=hidden_dim,
            kernel_size=temporal_kernel_size,
            dilation=temporal_dilation,
            dropout=temporal_dropout,
            use_layer_norm=use_layer_norm,
        )
        self.spatial = ResidualMultiSupportGraphBlock(
            hidden_dim=hidden_dim,
            support_names=spatial_supports,
            dropout=spatial_dropout,
            use_gate=spatial_gate,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected H_t sequence [B, L, N, D], got shape {tuple(x.shape)}")
        batch_size, seq_len, num_nodes, hidden_dim = x.shape
        hidden = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, hidden_dim, seq_len)
        hidden = self.temporal(hidden)
        hidden = hidden.reshape(batch_size, num_nodes, hidden_dim, seq_len).permute(0, 3, 1, 2)
        return self.spatial(hidden, graph)

class AlternatingSpatioTemporalEncoder(nn.Module):
    """Refine H_t with alternating temporal and anchor-support spatial interactions."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        temporal_kernel_size: int = 3,
        temporal_dilations: Optional[Tuple[int, ...]] = None,
        temporal_dropout: float = 0.0,
        spatial_supports: Tuple[str, ...] = ("identity", "forward", "backward"),
        spatial_dropout: float = 0.0,
        spatial_gate: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        num_layers = int(num_layers)
        if num_layers <= 0:
            raise ValueError("alternating_st_layers must be positive when the alternating encoder is enabled.")
        if temporal_dilations is None:
            temporal_dilations = tuple(2**layer_idx for layer_idx in range(num_layers))
        else:
            temporal_dilations = tuple(int(dilation) for dilation in temporal_dilations)
            if len(temporal_dilations) != num_layers:
                raise ValueError(
                    f"alternating_st_dilations length {len(temporal_dilations)} must match "
                    f"alternating_st_layers {num_layers}."
                )
        self.temporal_dilations = temporal_dilations
        self.spatial_supports = tuple(spatial_supports)
        self.layers = nn.ModuleList(
            [
                AlternatingSpatioTemporalBlock(
                    hidden_dim=hidden_dim,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_dilation=dilation,
                    temporal_dropout=temporal_dropout,
                    spatial_supports=self.spatial_supports,
                    spatial_dropout=spatial_dropout,
                    spatial_gate=spatial_gate,
                    use_layer_norm=use_layer_norm,
                )
                for dilation in self.temporal_dilations
            ]
        )

    def forward(self, temporal_state: torch.Tensor, anchor_support: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
        hidden = temporal_state
        layer_outputs = []
        for layer in self.layers:
            hidden = layer(hidden, anchor_support)
            layer_outputs.append(hidden)
        return hidden, {
            "alternating_st_encoder_status": "enabled_anchor_support_temporal_spatial_refiner",
            "alternating_st_dilations": list(self.temporal_dilations),
            "alternating_st_supports": list(self.spatial_supports),
            "H_t_alternating_layer_outputs": layer_outputs,
        }


