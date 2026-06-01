from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.temporal import GatedCausalTemporalConv

class HorizonContextEmbedding(nn.Module):
    """Learned forecast-horizon identity used by the residual decoder."""

    def __init__(self, horizon: int, horizon_dim: int, hidden_dim: int):
        super().__init__()
        self.horizon = int(horizon)
        self.embedding = nn.Embedding(self.horizon, horizon_dim)
        self.projection = nn.Linear(horizon_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, device: torch.device) -> torch.Tensor:
        horizon_ids = torch.arange(self.horizon, device=device)
        return self.projection(self.embedding(horizon_ids))

class HorizonAwareHistoryReadout(nn.Module):
    """Read horizon-specific structure states from the full history sequence."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        self.horizon_norm = nn.LayerNorm(hidden_dim)
        self.to_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        structure_sequence: torch.Tensor,
        horizon_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if structure_sequence.dim() != 4:
            raise ValueError(f"Expected H_struct_seq [B, L, N, D], got {tuple(structure_sequence.shape)}")
        if horizon_context.dim() != 2:
            raise ValueError(f"Expected horizon_context [H, D], got {tuple(horizon_context.shape)}")

        normalized_sequence = self.sequence_norm(structure_sequence)
        keys = self.to_key(normalized_sequence)
        values = self.to_value(normalized_sequence)
        queries = self.to_query(self.horizon_norm(horizon_context))

        scores = torch.einsum("hd,blnd->bhln", queries, keys) / (self.hidden_dim ** 0.5)
        weights = torch.softmax(scores, dim=2)
        context = torch.einsum("bhln,blnd->bhnd", weights, values)
        context = self.dropout(self.out_proj(context))

        last_state = structure_sequence[:, -1, :, :].unsqueeze(1)
        h_struct_future = last_state + context
        aux = {
            "horizon_history_weight": weights,
        }
        return h_struct_future, aux

class HorizonAwareResidualHead(nn.Module):
    """Shared one-step head applied to each future-context-conditioned residual state."""

    def __init__(self, hidden_dim: int, output_hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_hidden_dim)
        self.fc2 = nn.Linear(output_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected horizon-aware anchor state [B, H, N, D], got shape {tuple(x.shape)}")
        hidden = self.norm(x)
        hidden = self.dropout(F.relu(self.fc1(hidden)))
        prediction = self.fc2(hidden).squeeze(-1)
        return prediction.permute(0, 2, 1)

    def zero_init_output(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

class HorizonGatedForecastDecoder(nn.Module):
    """Causal horizon-axis decoder for residual trajectory refinement."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        dilations: Optional[Tuple[int, ...]] = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        num_layers = int(num_layers)
        if num_layers <= 0:
            raise ValueError("horizon_decoder_layers must be positive when horizon decoder is enabled.")
        if dilations is None:
            dilations = tuple(2**layer_idx for layer_idx in range(num_layers))
        else:
            dilations = tuple(int(dilation) for dilation in dilations)
            if len(dilations) != num_layers:
                raise ValueError(
                    f"horizon_decoder_dilations length {len(dilations)} must match "
                    f"horizon_decoder_layers {num_layers}."
                )
        self.dilations = dilations
        self.layers = nn.ModuleList(
            [
                GatedCausalTemporalConv(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for dilation in self.dilations
            ]
        )

    def forward(self, residual_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Union[str, torch.Tensor, list]]]:
        if residual_state.dim() != 4:
            raise ValueError(
                f"Expected horizon residual state [B, H, N, D], got shape {tuple(residual_state.shape)}"
            )
        batch_size, horizon, num_nodes, hidden_dim = residual_state.shape
        hidden = residual_state.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, hidden_dim, horizon)
        for layer in self.layers:
            hidden = layer(hidden)
        decoded = hidden.reshape(batch_size, num_nodes, hidden_dim, horizon).permute(0, 3, 1, 2)
        return decoded, {
            "horizon_forecast_decoder_status": "enabled_gated_causal_horizon_tcn",
            "horizon_forecast_decoder_dilations": list(self.dilations),
            "temporal_bias_state_pre_decoder": residual_state,
            "temporal_bias_state_decoded": decoded,
        }


