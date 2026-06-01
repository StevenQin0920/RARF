from __future__ import annotations

from typing import Optional, Tuple

import torch

from models.frozen_regime_anchor import FrozenRegimeAnchor
from models.rarf import AnchorConditionedInputs, RARF
from models.types import TensorAux
from models.anchor_conditioned_residual_correction import (
    TemporalBiasCorrectionBranch,
    TemporalBiasCorrectionEncoder,
)


ABLATION_VARIANTS = {
    "no-regime-anchor",
    "no-spatial-branch",
    "no-temporal-branch",
    "no-anchor-coordinate",
    "no-future-time",
    "no-fft-loss",
}


def normalize_ablation_variant(value: str) -> str:
    variant = str(value).strip().lower().replace("_", "-")
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant {value!r}. Valid variants: {sorted(ABLATION_VARIANTS)}")
    return variant


class NoAnchorCoordinateTemporalBiasEncoder(TemporalBiasCorrectionEncoder):
    """Temporal encoder ablation that consumes raw history instead of anchor-coordinate history."""

    def _setup_common(self, model_args: dict) -> None:
        super()._setup_common(model_args)
        self.anchor_residual_side_input_enabled = False
        self.anchor_input_mode = "raw_history_no_anchor_coordinate"
        self.anchor_coordinate_input_anchor_residual_leak = 0.0
        self.anchor_coordinate_input_strength = 0.0
        self.anchor_coordinate_input_schedule_progress = 0.0
        self.anchor_admission_gate_schedule_progress = 0.0

    def _setup_anchor_coordinate_inputs(self, model_args: dict) -> None:
        self.use_anchor_admission_gate = False
        self.anchor_admission_node_emb_dim = int(model_args.get("anchor_admission_node_emb_dim", 16))
        self.anchor_admission_residual_stat_dim = 10
        self.anchor_admission_node_embedding = None
        self.anchor_admission_gate = None

    def _residual_side_input_status(self) -> str:
        return "disabled_no_anchor_coordinate_ablation"

    def _construct_anchor_coordinate_inputs(
        self,
        history_data: torch.Tensor,
        history_tod: torch.Tensor,
        history_dow: torch.Tensor,
        reference_daily_history: torch.Tensor,
        reference_weekly_history: torch.Tensor,
        reference_daily_future: torch.Tensor,
        reference_weekly_future: torch.Tensor,
    ):
        del reference_daily_history, reference_weekly_history, reference_daily_future, reference_weekly_future
        traffic = history_data[:, :, :, 0]
        anchor_residual = torch.zeros_like(traffic)
        batch_size, _, num_nodes = traffic.shape
        admission_ratio = traffic.new_zeros((batch_size, num_nodes))
        aux: TensorAux = {
            "anchor_coordinate_input_status": "disabled_raw_history_ablation",
            "anchor_coordinate_input_mode": self.anchor_input_mode,
            "anchor_input_mode": self.anchor_input_mode,
            "anchor_coordinate_input_effective_strength": 0.0,
            "anchor_coordinate_input_schedule_progress": torch.tensor(
                0.0,
                device=history_data.device,
                dtype=history_data.dtype,
            ),
            "anchor_coordinate_A_history": torch.zeros_like(traffic),
            "anchor_coordinate_residual": anchor_residual,
            "anchor_residual_side_input_enabled": False,
            "anchor_residual_side_input_status": self._residual_side_input_status(),
            "anchor_coordinate_daily_history": torch.zeros_like(traffic),
            "anchor_coordinate_weekly_history": torch.zeros_like(traffic),
            "anchor_coordinate_recent_volatility": torch.zeros_like(admission_ratio),
            "anchor_admission_residual_stats_abs_mean": torch.zeros((), device=history_data.device),
            "anchor_admission_residual_scale_mean": torch.zeros((), device=history_data.device),
            "anchor_admission_gate_status": "disabled_no_anchor_coordinate_ablation",
            "anchor_admission_gate_ratio": admission_ratio,
            "anchor_admission_gate_ratio_mean": torch.zeros((), device=history_data.device),
            "anchor_admission_gate_ratio_std": torch.zeros((), device=history_data.device),
            "anchor_admission_gate_ratio_min": torch.zeros((), device=history_data.device),
            "anchor_admission_gate_ratio_max": torch.zeros((), device=history_data.device),
            "anchor_admission_gate_schedule_progress": torch.tensor(
                0.0,
                device=history_data.device,
                dtype=history_data.dtype,
            ),
            "corrected_anchor_daily_future": None,
            "corrected_anchor_weekly_future": None,
            "corrected_anchor_reference_future": None,
        }
        return self._anchor_inputs_type(anchor_history=history_data, anchor_residual=anchor_residual, aux=aux)

    @staticmethod
    def _anchor_inputs_type(anchor_history: torch.Tensor, anchor_residual: torch.Tensor, aux: TensorAux):
        from models.anchor_conditioned_residual_correction.temporal_bias_correction_encoder import AnchorCoordinateInputs

        return AnchorCoordinateInputs(
            anchor_history=anchor_history,
            anchor_residual=anchor_residual,
            aux=aux,
        )


class NoFutureTimeTemporalBiasBranch(TemporalBiasCorrectionBranch):
    """Temporal branch ablation that removes known future time indicators."""

    def _decode_state(
        self,
        structure_sequence: torch.Tensor,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, TensorAux]:
        del future_data
        batch_size = history_data.shape[0]
        horizon_context = self.horizon_context_embedding(history_data.device)
        structure_for_future, readout_aux = self.horizon_history_readout(structure_sequence, horizon_context)
        horizon_only_context = self.horizon_context_projection(horizon_context).unsqueeze(0).expand(batch_size, -1, -1)
        future_delta = self.future_context_state_projection(self.future_context_state_norm(horizon_only_context))
        future_state = structure_for_future + future_delta.unsqueeze(2)
        decoded_state, decoder_aux = self.horizon_forecast_decoder(future_state)
        zero_ids = torch.zeros(batch_size, self.horizon, device=history_data.device, dtype=torch.long)
        aux: TensorAux = {
            "horizon_context": horizon_context,
            "horizon_history_readout_status": "enabled_attention_over_H_struct_seq",
            "h_struct_future": structure_for_future,
            "future_tod": zero_ids,
            "future_dow": zero_ids,
            "future_weekend": zero_ids,
            "future_holiday": zero_ids,
            "future_context_channels_used": [],
            "future_weekend_context_feature_enabled": False,
            "future_holiday_context_feature_enabled": False,
            "future_speed_channel_consumed": False,
            "future_context_status": {
                "future_context_alignment": "disabled_no_future_time_ablation",
                "future_tod_source": "disabled",
                "future_dow_source": "disabled",
                "future_weekend_source": "disabled",
                "future_holiday_source": "disabled",
                "future_weekend_context_feature_enabled": False,
                "future_holiday_context_feature_enabled": False,
                "future_speed_channel_consumed": False,
            },
            "future_context_alignment": "horizon_identity_only",
            "future_context_query": horizon_only_context,
            "temporal_bias_state_pre_decoder": future_state,
            "temporal_bias_state": decoded_state,
            "future_context_conditioning_status": "disabled_no_future_time_ablation",
            "future_context_node_conditioning": "none",
            "future_context_node_conditioning_status": "disabled",
            **readout_aux,
            **decoder_aux,
        }
        return decoded_state, aux


class RARFAblation(RARF):
    """Isolated RARF ablation variants for paper experiments."""

    def __init__(self, *, ablation_variant: str, **model_args: object) -> None:
        self.ablation_variant = normalize_ablation_variant(ablation_variant)
        self._ablation_model_args = dict(model_args)
        super().__init__(**model_args)
        if self.ablation_variant == "no-anchor-coordinate":
            self.temporal_bias_encoder = NoAnchorCoordinateTemporalBiasEncoder(**model_args)
            self.temporal_bias_encoder.reset_paper_parameters()
        if self.ablation_variant == "no-future-time":
            self.temporal_bias_branch = NoFutureTimeTemporalBiasBranch(**model_args)

    def _build_frozen_anchor(self, model_args: dict) -> FrozenRegimeAnchor:
        if getattr(self, "ablation_variant", None) != "no-regime-anchor":
            return super()._build_frozen_anchor(model_args)
        daily_zeros = torch.zeros(self.tod_vocab_size, self.num_nodes, dtype=torch.float32)
        weekly_zeros = torch.zeros(self.tod_vocab_size * self.dow_vocab_size, self.num_nodes, dtype=torch.float32)
        return FrozenRegimeAnchor(
            num_nodes=self.num_nodes,
            tod_vocab_size=self.tod_vocab_size,
            dow_vocab_size=self.dow_vocab_size,
            horizon=self.horizon,
            daily_init=daily_zeros,
            weekly_init=weekly_zeros,
            daily_weight=float(model_args.get("regime_daily_weight", 1.0)),
            weekly_weight=float(model_args.get("regime_weekly_weight", 1.0)),
            spatial_bias_graph_adj=model_args.get("spatial_bias_graph_adj", model_args.get("prior_adj")),
        )

    def _anchor_residual_components(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor],
    ) -> AnchorConditionedInputs:
        if self.ablation_variant != "no-spatial-branch":
            return super()._anchor_residual_components(history_data, future_data)

        history_tod, history_dow = self._temporal_id_sequences(history_data)
        future_tod, future_dow = self._future_temporal_id_sequences(history_data, future_data)
        daily_a0_history, weekly_a0_history = self.frozen_anchor.lookup_a0_components(history_tod, history_dow)
        daily_a0_future, weekly_a0_future = self.frozen_anchor.lookup_a0_components(future_tod, future_dow)
        daily_zero_history = torch.zeros_like(daily_a0_history)
        weekly_zero_history = torch.zeros_like(weekly_a0_history)
        daily_zero_future = torch.zeros_like(daily_a0_future)
        weekly_zero_future = torch.zeros_like(weekly_a0_future)
        A0_history = daily_a0_history + weekly_a0_history
        A0_future = (daily_a0_future + weekly_a0_future).transpose(1, 2)
        R_spatial_history = daily_zero_history + weekly_zero_history
        R_spatial_future = (daily_zero_future + weekly_zero_future).transpose(1, 2)
        return AnchorConditionedInputs(
            history_tod=history_tod,
            history_dow=history_dow,
            future_tod=future_tod,
            future_dow=future_dow,
            A0_history=A0_history,
            A0_future=A0_future,
            R_spatial_history=R_spatial_history,
            R_spatial_future=R_spatial_future,
            reference_daily_history=daily_a0_history,
            reference_weekly_history=weekly_a0_history,
            reference_daily_future=daily_a0_future.transpose(1, 2),
            reference_weekly_future=weekly_a0_future.transpose(1, 2),
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor] = None,
        return_graphs: bool = False,
        return_aux: bool = False,
        aux_mode: str = "full",
        aux_keep_keys: Optional[Tuple[str, ...]] = None,
    ):
        if self.ablation_variant != "no-temporal-branch":
            return super().forward(
                history_data,
                future_data=future_data,
                return_graphs=return_graphs,
                return_aux=return_aux,
                aux_mode=aux_mode,
                aux_keep_keys=aux_keep_keys,
            )

        correction_inputs = self._anchor_residual_components(history_data, future_data)
        temporal_bias_correction = torch.zeros_like(correction_inputs.R_spatial_future)
        anchor_conditioned_residual_correction = correction_inputs.R_spatial_future
        y_hat = correction_inputs.A0_future + anchor_conditioned_residual_correction
        if return_graphs or return_aux:
            raw_aux: TensorAux = {
                "A0_future": correction_inputs.A0_future,
                "spatial_bias_correction": correction_inputs.R_spatial_future,
                "temporal_bias_correction": temporal_bias_correction,
                "anchor_conditioned_residual_correction": anchor_conditioned_residual_correction,
                "regime_daily_weight": self.frozen_anchor.daily_weight,
                "regime_weekly_weight": self.frozen_anchor.weekly_weight,
                "forecast_mode": f"{self.forecast_mode}_ablation_no_temporal_branch",
                "temporal_bias_branch_status": "disabled_no_temporal_branch_ablation",
            }
            aux = self._public_aux(raw_aux, include_graphs=return_graphs)
            if aux_mode == "train_minimal":
                aux = self._minimal_train_aux(aux, keep_tensor_keys=aux_keep_keys)
            return y_hat, aux
        return y_hat
