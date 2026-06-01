from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .heads import HorizonAwareHistoryReadout, HorizonAwareResidualHead, HorizonContextEmbedding, HorizonGatedForecastDecoder
from .future_context import FutureContextEncoder
from ..types import TensorAux


class TemporalBiasCorrectionBranch(nn.Module):
    """Decode future Temporal-Bias correction from encoded history structure.

    The encoder already placed history in anchor coordinates. This branch only
    reads the encoded structure sequence, injects known future time context, and
    predicts R_temporal with shape [B, N, H].
    """

    def __init__(self, **model_args: object) -> None:
        super().__init__()
        self._setup_common(model_args)
        self._setup_future_context_state(model_args)
        self._setup_residual_predictor(model_args)

    def _setup_common(self, model_args: dict) -> None:
        self.num_nodes = int(model_args["num_nodes"])
        self.hidden_dim = int(model_args.get("num_hidden", model_args.get("hidden_dim", 64)))
        self.time_emb_dim = int(model_args.get("time_emb_dim", 16))
        self.output_hidden_dim = int(model_args.get("output_hidden_dim", 256))
        self.horizon = int(model_args.get("horizon", model_args.get("seq_length", 12)))
        self.tod_vocab_size = int(model_args.get("tod_vocab_size", 288))
        self.dow_vocab_size = int(model_args.get("dow_vocab_size", 7))
        self.weekend_vocab_size = int(model_args.get("weekend_vocab_size", 2))
        self.holiday_vocab_size = int(model_args.get("holiday_vocab_size", 2))
        self.future_context_tod_channel_index = int(model_args.get("future_context_tod_channel_index", 1))
        self.future_context_dow_channel_index = int(model_args.get("future_context_dow_channel_index", 2))
        self.future_context_weekend_channel_index = int(model_args.get("future_context_weekend_channel_index", 3))
        self.future_context_holiday_channel_index = int(model_args.get("future_context_holiday_channel_index", 4))
        self.use_future_weekend_context_feature = bool(model_args.get("use_future_weekend_context_feature", True))
        self.use_future_holiday_context_feature = bool(model_args.get("use_future_holiday_context_feature", True))
        self.holiday_mode = str(model_args.get("holiday_mode", "input_channel"))
        self.dropout = float(model_args.get("dropout", 0.1))
        self.residual_decoder_dropout = float(model_args.get("residual_decoder_dropout", self.dropout))

    def _setup_future_context_state(self, model_args: dict) -> None:
        self.horizon_context_embedding = HorizonContextEmbedding(
            horizon=self.horizon,
            horizon_dim=int(model_args.get("horizon_emb_dim", self.time_emb_dim)),
            hidden_dim=self.hidden_dim,
        )
        self.horizon_history_readout = HorizonAwareHistoryReadout(
            hidden_dim=self.hidden_dim,
            dropout=float(model_args.get("horizon_readout_dropout", self.residual_decoder_dropout)),
        )
        self.future_context_hidden_dim = int(model_args.get("future_context_hidden_dim", self.hidden_dim))
        self.future_context_encoder = FutureContextEncoder(
            tod_vocab_size=self.tod_vocab_size,
            dow_vocab_size=self.dow_vocab_size,
            weekend_vocab_size=self.weekend_vocab_size,
            holiday_vocab_size=self.holiday_vocab_size,
            tod_emb_dim=int(model_args.get("tod_emb_dim", 16)),
            dow_emb_dim=int(model_args.get("dow_emb_dim", 8)),
            weekend_emb_dim=int(model_args.get("weekend_emb_dim", 4)),
            holiday_emb_dim=int(model_args.get("holiday_emb_dim", 4)),
            output_dim=self.future_context_hidden_dim,
            use_weekend=self.use_future_weekend_context_feature,
            use_holiday=self.use_future_holiday_context_feature,
        )
        self.horizon_context_projection = nn.Linear(self.hidden_dim, self.future_context_hidden_dim)
        self.future_context_state_norm = nn.LayerNorm(self.future_context_hidden_dim)
        self.future_context_state_projection = nn.Linear(self.future_context_hidden_dim, self.hidden_dim)
        nn.init.zeros_(self.future_context_state_projection.weight)
        nn.init.zeros_(self.future_context_state_projection.bias)

    def _setup_residual_predictor(self, model_args: dict) -> None:
        self.horizon_forecast_decoder = HorizonGatedForecastDecoder(
            hidden_dim=self.hidden_dim,
            num_layers=int(model_args.get("horizon_decoder_layers", 2)),
            kernel_size=int(model_args.get("horizon_decoder_kernel_size", 3)),
            dropout=float(model_args.get("horizon_decoder_dropout", self.residual_decoder_dropout)),
            dilations=(
                tuple(model_args["horizon_decoder_dilations"])
                if model_args.get("horizon_decoder_dilations") is not None
                else None
            ),
            use_layer_norm=bool(model_args.get("horizon_decoder_layer_norm", True)),
        )
        self.residual_head = HorizonAwareResidualHead(
            hidden_dim=self.hidden_dim,
            output_hidden_dim=self.output_hidden_dim,
            dropout=self.residual_decoder_dropout,
        )

    def _future_context_inputs(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        if future_data is None:
            raise ValueError("RARF requires future_data temporal channels.")
        if future_data.dim() != 4:
            raise ValueError(f"future_data must be [B,H,N,C], got {tuple(future_data.shape)}.")
        if future_data.shape[0] != history_data.shape[0] or future_data.shape[2] != self.num_nodes:
            raise ValueError("future_data batch and node count must match history_data.")
        if future_data.shape[1] != self.horizon:
            raise ValueError(f"future_data horizon {future_data.shape[1]} != configured horizon {self.horizon}.")
        required_idx = max(self.future_context_tod_channel_index, self.future_context_dow_channel_index)
        if future_data.shape[-1] <= required_idx:
            raise ValueError("future_data lacks required TOD/DOW channels.")

        future_tod = torch.floor(
            future_data[:, :, 0, self.future_context_tod_channel_index] * self.tod_vocab_size
        ).long()
        future_tod = future_tod.clamp(0, self.tod_vocab_size - 1)
        future_dow = future_data[:, :, 0, self.future_context_dow_channel_index].long().clamp(
            0,
            self.dow_vocab_size - 1,
        )
        if self.use_future_weekend_context_feature and future_data.shape[-1] > self.future_context_weekend_channel_index:
            future_weekend = future_data[:, :, 0, self.future_context_weekend_channel_index].long().clamp(
                0,
                self.weekend_vocab_size - 1,
            )
            weekend_source = f"future_data_channel_{self.future_context_weekend_channel_index}"
        else:
            future_weekend = torch.zeros_like(future_tod)
            weekend_source = "placeholder_zero_unavailable"
        if (
            self.use_future_holiday_context_feature
            and self.holiday_mode == "input_channel"
            and future_data.shape[-1] > self.future_context_holiday_channel_index
        ):
            future_holiday = future_data[:, :, 0, self.future_context_holiday_channel_index].long().clamp(
                0,
                self.holiday_vocab_size - 1,
            )
            holiday_source = f"future_data_channel_{self.future_context_holiday_channel_index}"
        else:
            future_holiday = torch.zeros_like(future_tod)
            holiday_source = "placeholder_zero_unavailable"
        return future_tod, future_dow, future_weekend, future_holiday, {
            "future_context_alignment": "future_horizon_aware",
            "future_tod_source": f"future_data_channel_{self.future_context_tod_channel_index}",
            "future_dow_source": f"future_data_channel_{self.future_context_dow_channel_index}",
            "future_weekend_source": weekend_source,
            "future_holiday_source": holiday_source,
            "future_weekend_context_feature_enabled": self.use_future_weekend_context_feature,
            "future_holiday_context_feature_enabled": self.use_future_holiday_context_feature,
            "future_speed_channel_consumed": False,
        }

    def _decode_state(
        self,
        structure_sequence: torch.Tensor,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, TensorAux]:
        horizon_context = self.horizon_context_embedding(history_data.device)
        structure_for_future, readout_aux = self.horizon_history_readout(structure_sequence, horizon_context)
        future_tod, future_dow, future_weekend, future_holiday, future_context_status = self._future_context_inputs(
            history_data,
            future_data,
        )
        future_context = self.future_context_encoder(future_tod, future_dow, future_weekend, future_holiday)
        future_context = future_context + self.horizon_context_projection(horizon_context).unsqueeze(0)
        future_delta = self.future_context_state_projection(self.future_context_state_norm(future_context))
        future_state = structure_for_future + future_delta.unsqueeze(2)
        decoded_state, decoder_aux = self.horizon_forecast_decoder(future_state)
        aux: TensorAux = {
            "horizon_context": horizon_context,
            "horizon_history_readout_status": "enabled_attention_over_H_struct_seq",
            "h_struct_future": structure_for_future,
            "future_tod": future_tod,
            "future_dow": future_dow,
            "future_weekend": future_weekend,
            "future_holiday": future_holiday,
            "future_context_channels_used": [
                self.future_context_tod_channel_index,
                self.future_context_dow_channel_index,
                self.future_context_weekend_channel_index,
                self.future_context_holiday_channel_index,
            ],
            "future_weekend_context_feature_enabled": self.use_future_weekend_context_feature,
            "future_holiday_context_feature_enabled": self.use_future_holiday_context_feature,
            "future_speed_channel_consumed": False,
            "future_context_status": future_context_status,
            "future_context_alignment": "future_time_mlp_add",
            "future_context_query": future_context,
            "temporal_bias_state_pre_decoder": future_state,
            "temporal_bias_state": decoded_state,
            "future_context_conditioning_status": "enabled_future_time_mlp_add",
            "future_context_node_conditioning": "global",
            "future_context_node_conditioning_status": "global_broadcast",
            **readout_aux,
            **decoder_aux,
        }
        return decoded_state, aux

    def forward(
        self,
        structure_sequence: torch.Tensor,
        temporal_state: torch.Tensor,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor],
        *,
        anchor_residual: Optional[torch.Tensor],
    ) -> Dict[str, object]:
        del temporal_state, anchor_residual
        decoded_state, state_aux = self._decode_state(structure_sequence, history_data, future_data)
        temporal_bias_correction = self.residual_head(decoded_state)
        zero_residual = torch.zeros_like(temporal_bias_correction)
        aux: TensorAux = {
            "temporal_bias_correction": temporal_bias_correction,
            "temporal_bias_aux_disabled": zero_residual,
            "hidden_history_residual_status": "disabled",
            "frozen_anchor_status": "used_as_forecast_base",
            "anchor_residual_side_input_enabled": True,
            **state_aux,
        }
        return {
            "temporal_bias_correction": temporal_bias_correction,
            "decoded_state": decoded_state,
            "aux": aux,
        }
