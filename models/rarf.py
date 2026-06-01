from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .frozen_regime_anchor import FrozenRegimeAnchor
from .types import TensorAux
from .anchor_conditioned_residual_correction import (
    TemporalBiasCorrectionBranch,
    TemporalBiasCorrectionEncoder,
)


@dataclass(frozen=True)
class AnchorConditionedInputs:
    """Intermediate tensors shared by Spatial-Bias and Temporal-Bias branches.

    Shape contract:
        history_data: [B, T, N, C]
        future_data: [B, H, N, C]
        A0_future, R_spatial_future, R_temporal, Y_hat: [B, N, H]
        reference_*_history: [B, T, N]
        reference_*_future: [B, H, N]
    """

    history_tod: torch.Tensor
    history_dow: torch.Tensor
    future_tod: torch.Tensor
    future_dow: torch.Tensor
    A0_history: torch.Tensor
    A0_future: torch.Tensor
    R_spatial_history: torch.Tensor
    R_spatial_future: torch.Tensor
    reference_daily_history: torch.Tensor
    reference_weekly_history: torch.Tensor
    reference_daily_future: torch.Tensor
    reference_weekly_future: torch.Tensor


class RARF(nn.Module):
    """Regime-Anchored Residual Forecasting.

    Released formula:

        Y_hat = A0_future + R_corr
        R_corr = R_spatial + R_temporal
    """

    PUBLIC_TENSOR_AUX_KEYS = {
        "A0_future",
        "spatial_bias_correction",
        "temporal_bias_correction",
        "anchor_conditioned_residual_correction",
    }
    GRAPH_AUX_KEYS = {"prior_adj", "physical_adj", "residual_support_graph"}
    PUBLIC_STATUS_AUX_KEYS = {
        "anchor_coordinate_input_status",
        "anchor_coordinate_input_mode",
        "anchor_input_mode",
        "anchor_admission_gate_status",
        "anchor_residual_side_input_status",
        "input_missing_mask_status",
        "input_missing_mask_policy",
        "frozen_anchor_status",
        "structure_encoder_status",
        "forecast_mode",
        "future_context_conditioning_status",
        "horizon_forecast_decoder_status",
        "target_zero_is_valid",
        "temporal_encoder_type",
    }
    PRIVATE_AUX_KEYS = {"model_variant"}

    def __init__(self, **model_args: object) -> None:
        super().__init__()
        model_variant = str(model_args.get("model_variant", "rarf"))
        if model_variant != "rarf":
            raise ValueError("RARF only supports model_variant='rarf'.")
        self.model_variant = model_variant
        self.forecast_mode = "anchor_conditioned_residual_correction"
        self._setup_common(model_args)
        self.frozen_anchor = self._build_frozen_anchor(model_args)
        self.temporal_bias_encoder = TemporalBiasCorrectionEncoder(**model_args)
        self.temporal_bias_branch = TemporalBiasCorrectionBranch(**model_args)
        self.temporal_bias_encoder.reset_paper_parameters()

    def _setup_common(self, model_args: dict) -> None:
        self.num_nodes = int(model_args["num_nodes"])
        self.num_feat = int(model_args.get("num_feat", 1))
        self.horizon = int(model_args.get("horizon", model_args.get("seq_length", 12)))
        self.tod_vocab_size = int(model_args.get("tod_vocab_size", 288))
        self.dow_vocab_size = int(model_args.get("dow_vocab_size", 7))
        self.future_context_tod_channel_index = int(model_args.get("future_context_tod_channel_index", 1))
        self.future_context_dow_channel_index = int(model_args.get("future_context_dow_channel_index", 2))

    def _build_frozen_anchor(self, model_args: dict) -> FrozenRegimeAnchor:
        return FrozenRegimeAnchor(
            num_nodes=self.num_nodes,
            tod_vocab_size=self.tod_vocab_size,
            dow_vocab_size=self.dow_vocab_size,
            horizon=self.horizon,
            daily_init_path=model_args.get("regime_anchor_field_daily_path"),
            weekly_init_path=model_args.get("regime_anchor_field_weekly_path"),
            daily_init=model_args.get("regime_daily_init"),
            weekly_init=model_args.get("regime_weekly_init"),
            daily_weight=float(model_args.get("regime_daily_weight", 1.0)),
            weekly_weight=float(model_args.get("regime_weekly_weight", 1.0)),
            spatial_bias_graph_adj=model_args.get("spatial_bias_graph_adj", model_args.get("prior_adj")),
        )

    def set_anchor_coordinate_input_schedule(self, current_step: int, warmup_steps: int) -> None:
        self.temporal_bias_encoder.set_anchor_coordinate_input_schedule(current_step, warmup_steps)

    def set_anchor_admission_gate_schedule(self, current_step: int, warmup_steps: int) -> None:
        self.temporal_bias_encoder.set_anchor_admission_gate_schedule(current_step, warmup_steps)

    def _temporal_id_sequences(self, history_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_idx = self.num_feat
        day_idx = self.num_feat + 1
        batch_size, seq_len, num_nodes, _ = history_data.shape
        if history_data.shape[-1] > time_idx:
            time_ids = torch.floor(history_data[:, :, :, time_idx] * self.tod_vocab_size).long()
            time_ids = time_ids.clamp(0, self.tod_vocab_size - 1)
        else:
            time_ids = torch.zeros(batch_size, seq_len, num_nodes, dtype=torch.long, device=history_data.device)
        if history_data.shape[-1] > day_idx:
            day_ids = history_data[:, :, :, day_idx].long().clamp(0, self.dow_vocab_size - 1)
        else:
            day_ids = torch.zeros(batch_size, seq_len, num_nodes, dtype=torch.long, device=history_data.device)
        return time_ids, day_ids

    def _future_temporal_id_sequences(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return future_tod, future_dow

    def _anchor_residual_components(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor],
    ) -> AnchorConditionedInputs:
        history_tod, history_dow = self._temporal_id_sequences(history_data)
        future_tod, future_dow = self._future_temporal_id_sequences(history_data, future_data)

        daily_a0_history, weekly_a0_history = self.frozen_anchor.lookup_a0_components(history_tod, history_dow)
        daily_a0_future, weekly_a0_future = self.frozen_anchor.lookup_a0_components(future_tod, future_dow)

        daily_spatial_history, weekly_spatial_history = self.frozen_anchor.lookup_spatial_bias_components(
            history_tod,
            history_dow,
        )
        daily_spatial_future, weekly_spatial_future = self.frozen_anchor.lookup_spatial_bias_components(
            future_tod,
            future_dow,
        )

        reference_daily_history = daily_a0_history + daily_spatial_history
        reference_weekly_history = weekly_a0_history + weekly_spatial_history
        reference_daily_future = daily_a0_future + daily_spatial_future
        reference_weekly_future = weekly_a0_future + weekly_spatial_future

        A0_history = daily_a0_history + weekly_a0_history
        R_spatial_history = daily_spatial_history + weekly_spatial_history
        A0_future = (daily_a0_future + weekly_a0_future).transpose(1, 2)
        R_spatial_future = (daily_spatial_future + weekly_spatial_future).transpose(1, 2)

        return AnchorConditionedInputs(
            history_tod=history_tod,
            history_dow=history_dow,
            future_tod=future_tod,
            future_dow=future_dow,
            A0_history=A0_history,
            A0_future=A0_future,
            R_spatial_history=R_spatial_history,
            R_spatial_future=R_spatial_future,
            reference_daily_history=reference_daily_history,
            reference_weekly_history=reference_weekly_history,
            reference_daily_future=reference_daily_future.transpose(1, 2),
            reference_weekly_future=reference_weekly_future.transpose(1, 2),
        )

    @staticmethod
    def _minimal_train_aux(aux: TensorAux, keep_tensor_keys: Optional[Tuple[str, ...]]) -> TensorAux:
        keep = set(keep_tensor_keys or ())
        keep.update({"A0_future", "anchor_conditioned_residual_correction"})
        minimal: TensorAux = {}
        for key, value in aux.items():
            if key in keep:
                minimal[key] = value
        return minimal

    @classmethod
    def _public_aux(cls, aux: TensorAux, *, include_graphs: bool) -> TensorAux:
        public: TensorAux = {}
        tensor_keep = set(cls.PUBLIC_TENSOR_AUX_KEYS)
        if include_graphs:
            tensor_keep.update(cls.GRAPH_AUX_KEYS)
        for key, value in aux.items():
            if key in cls.PRIVATE_AUX_KEYS:
                continue
            if key in tensor_keep or key in cls.PUBLIC_STATUS_AUX_KEYS:
                public[key] = value
        return public

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor] = None,
        return_graphs: bool = False,
        return_aux: bool = False,
        aux_mode: str = "full",
        aux_keep_keys: Optional[Tuple[str, ...]] = None,
    ):
        aux_mode = str(aux_mode).lower()
        if aux_mode not in {"full", "train_minimal"}:
            raise ValueError(f"aux_mode must be 'full' or 'train_minimal', got {aux_mode!r}.")

        correction_inputs = self._anchor_residual_components(history_data, future_data)
        encoded = self.temporal_bias_encoder(
            history_data,
            history_tod=correction_inputs.history_tod,
            history_dow=correction_inputs.history_dow,
            reference_daily_history=correction_inputs.reference_daily_history,
            reference_weekly_history=correction_inputs.reference_weekly_history,
            reference_daily_future=correction_inputs.reference_daily_future,
            reference_weekly_future=correction_inputs.reference_weekly_future,
        )
        decoded = self.temporal_bias_branch(
            encoded["structure_sequence"],
            encoded["temporal_state"],
            history_data,
            future_data,
            anchor_residual=encoded["anchor_residual"],
        )
        temporal_bias_correction = decoded["temporal_bias_correction"]
        if not isinstance(temporal_bias_correction, torch.Tensor):
            raise RuntimeError("RARF Temporal-Bias Correction Branch did not produce temporal_bias_correction.")
        anchor_conditioned_residual_correction = correction_inputs.R_spatial_future + temporal_bias_correction
        Y_hat = correction_inputs.A0_future + anchor_conditioned_residual_correction

        if return_graphs or return_aux:
            raw_aux: TensorAux = {
                **encoded["aux"],
                **decoded["aux"],
                "A0_future": correction_inputs.A0_future,
                "spatial_bias_correction": correction_inputs.R_spatial_future,
                "temporal_bias_correction": temporal_bias_correction,
                "anchor_conditioned_residual_correction": anchor_conditioned_residual_correction,
                "regime_daily_weight": self.frozen_anchor.daily_weight,
                "regime_weekly_weight": self.frozen_anchor.weekly_weight,
                "forecast_mode": self.forecast_mode,
            }
            aux = self._public_aux(raw_aux, include_graphs=return_graphs)
            if aux_mode == "train_minimal":
                aux = self._minimal_train_aux(aux, keep_tensor_keys=aux_keep_keys)
            return Y_hat, aux
        return Y_hat
