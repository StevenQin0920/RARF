from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .anchor_admission import AnchorAdmissionGate
from .backbone.common import _as_float_tensor, row_normalize
from .backbone.temporal import AlternatingSpatioTemporalEncoder, TemporalConvEncoder
from ..types import TensorAux


@dataclass(frozen=True)
class AnchorCoordinateInputs:
    """Anchor-coordinate history prepared for temporal encoding."""

    anchor_history: torch.Tensor
    anchor_residual: torch.Tensor
    aux: TensorAux


class TemporalBiasCorrectionEncoder(nn.Module):
    """Encode history around the anchor-conditioned reference.

    Forward stages:
        1. Build an anchor-coordinate history signal from X_hist and A0 + R_spatial.
        2. Encode that signal with temporal and physical-graph structure encoders.
        3. Return a structure sequence consumed by the Temporal-Bias Correction Branch.

    Main shapes:
        history_data: [B, T, N, C]
        reference_*_history: [B, T, N]
        reference_*_future: [B, H, N]
        structure_sequence: [B, T, N, hidden_dim]
    """

    def __init__(self, **model_args: object) -> None:
        super().__init__()
        self._setup_common(model_args)
        self._setup_graph_buffers(model_args)
        self._setup_temporal_path(model_args)
        self._setup_anchor_coordinate_inputs(model_args)

    def _setup_common(self, model_args: dict) -> None:
        self.epsilon = 1e-7
        self.num_nodes = int(model_args["num_nodes"])
        self.num_feat = int(model_args.get("num_feat", 1))
        self.hidden_dim = int(model_args.get("num_hidden", model_args.get("hidden_dim", 64)))
        self.time_emb_dim = int(model_args.get("time_emb_dim", 16))
        self.horizon = int(model_args.get("horizon", model_args.get("seq_length", 12)))
        self.history_length = int(model_args.get("history_length", self.horizon))
        self.tod_vocab_size = int(model_args.get("tod_vocab_size", 288))
        self.dow_vocab_size = int(model_args.get("dow_vocab_size", 7))
        self.dropout = float(model_args.get("dropout", 0.1))
        self.temporal_dropout = float(model_args.get("temporal_dropout", self.dropout))
        self.graph_dropout = float(model_args.get("graph_dropout", self.dropout))
        self.residual_encoder_dropout = float(model_args.get("residual_encoder_dropout", self.dropout))

        self.use_input_missing_mask = bool(model_args.get("use_input_missing_mask", True))
        self.target_zero_is_valid = bool(model_args.get("target_zero_is_valid", False))
        self.input_missing_mask_policy = str(model_args.get("input_missing_mask_policy", "zero_as_missing"))
        self.input_missing_fill_value = float(model_args.get("input_missing_fill_value", 0.0))
        self.missing_zero_value = float(model_args.get("missing_zero_value", 0.0))
        self.missing_mask_tolerance = float(model_args.get("missing_mask_tolerance", 1e-5))

        self.anchor_residual_side_input_enabled = True
        self.anchor_residual_input_detach = bool(model_args.get("anchor_residual_input_detach", True))
        self.anchor_input_mode = "learned_alpha_with_residual_side_input"

    def _setup_graph_buffers(self, model_args: dict) -> None:
        prior_adj = row_normalize(torch.clamp(_as_float_tensor(model_args["prior_adj"]), min=0.0), self.epsilon)
        physical_adj = row_normalize(torch.clamp(_as_float_tensor(model_args["physical_adj"]), min=0.0), self.epsilon)
        if tuple(prior_adj.shape) != (self.num_nodes, self.num_nodes):
            raise ValueError("prior_adj shape does not match num_nodes.")
        if tuple(physical_adj.shape) != (self.num_nodes, self.num_nodes):
            raise ValueError("physical_adj shape does not match num_nodes.")
        self.register_buffer("prior_adj", prior_adj)
        self.register_buffer("physical_adj", physical_adj)
        self.anchor_support_graph = None

    def _setup_temporal_path(self, model_args: dict) -> None:
        temporal_input_dim = self.num_feat
        if self.use_input_missing_mask:
            temporal_input_dim += 1
        if self.anchor_residual_side_input_enabled:
            temporal_input_dim += 1

        self.time_in_day_embedding = nn.Embedding(self.tod_vocab_size, self.time_emb_dim)
        self.day_in_week_embedding = nn.Embedding(self.dow_vocab_size, self.time_emb_dim)
        self.temporal_encoder_type = str(model_args.get("temporal_encoder_type", "gated_tcn"))
        self.temporal_dilations = model_args.get("temporal_dilations")
        self.temporal_layer_norm = bool(model_args.get("temporal_layer_norm", True))
        self.temporal_encoder = TemporalConvEncoder(
            input_dim=temporal_input_dim,
            hidden_dim=self.hidden_dim,
            time_emb_dim=self.time_emb_dim,
            num_layers=int(model_args.get("temporal_layers", 4)),
            kernel_size=int(model_args.get("temporal_kernel_size", 3)),
            dropout=self.temporal_dropout,
            encoder_type=self.temporal_encoder_type,
            dilations=tuple(self.temporal_dilations) if self.temporal_dilations is not None else None,
            use_layer_norm=self.temporal_layer_norm,
        )
        if not bool(model_args.get("use_alternating_st_encoder", True)):
            raise ValueError("RARF requires use_alternating_st_encoder=True.")
        self.alternating_st_encoder = AlternatingSpatioTemporalEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=int(model_args.get("alternating_st_layers", 2)),
            temporal_kernel_size=int(model_args.get("alternating_st_kernel_size", 3)),
            temporal_dilations=(
                tuple(model_args["alternating_st_dilations"])
                if model_args.get("alternating_st_dilations") is not None
                else None
            ),
            temporal_dropout=float(model_args.get("alternating_st_temporal_dropout", self.temporal_dropout)),
            spatial_supports=tuple(model_args.get("alternating_st_supports", ("identity", "forward", "backward"))),
            spatial_dropout=float(model_args.get("alternating_st_spatial_dropout", self.graph_dropout)),
            spatial_gate=bool(model_args.get("alternating_st_gate", True)),
            use_layer_norm=bool(model_args.get("alternating_st_layer_norm", True)),
        )

    def _setup_anchor_coordinate_inputs(self, model_args: dict) -> None:
        self.use_anchor_admission_gate = bool(model_args.get("use_anchor_admission_gate", True))
        self.anchor_coordinate_input_anchor_residual_leak = float(
            model_args.get("anchor_coordinate_input_anchor_residual_leak", 0.7)
        )
        self.anchor_coordinate_input_strength = float(model_args.get("anchor_coordinate_input_strength", 1.0))
        self.anchor_coordinate_input_schedule_progress = 1.0
        self.anchor_admission_gate_schedule_progress = 1.0
        self.anchor_admission_node_emb_dim = int(model_args.get("anchor_admission_node_emb_dim", 16))
        self.anchor_admission_residual_stat_dim = 10
        if self.anchor_admission_node_emb_dim <= 0:
            raise ValueError("anchor_admission_node_emb_dim must be positive.")
        gate_input_dim = (
            self.anchor_admission_node_emb_dim
            + 2 * self.time_emb_dim
            + self.anchor_admission_residual_stat_dim
            + 4
        )
        self.anchor_admission_node_embedding = (
            nn.Embedding(self.num_nodes, self.anchor_admission_node_emb_dim)
            if self.use_anchor_admission_gate
            else None
        )
        if self.anchor_admission_node_embedding is not None:
            nn.init.xavier_uniform_(self.anchor_admission_node_embedding.weight)
        self.anchor_admission_gate = (
            AnchorAdmissionGate(
                gate_input_dim,
                int(model_args.get("anchor_admission_gate_hidden_dim", 64)),
                r_min=float(model_args.get("anchor_admission_gate_ratio_min", 0.0)),
                r_max=float(model_args.get("anchor_admission_gate_ratio_max", 0.8)),
                r_init=float(model_args.get("anchor_admission_gate_ratio_init", 0.2)),
            )
            if self.use_anchor_admission_gate
            else None
        )

    def _residual_side_input_status(self) -> str:
        return "enabled_residual_side_input"

    def reset_paper_parameters(self) -> None:
        nn.init.xavier_uniform_(self.time_in_day_embedding.weight)
        nn.init.xavier_uniform_(self.day_in_week_embedding.weight)

    def set_anchor_coordinate_input_schedule(self, current_step: int, warmup_steps: int) -> None:
        if warmup_steps <= 0:
            self.anchor_coordinate_input_schedule_progress = 1.0
        else:
            self.anchor_coordinate_input_schedule_progress = min(
                max(float(current_step) / float(warmup_steps), 0.0),
                1.0,
            )

    def set_anchor_admission_gate_schedule(self, current_step: int, warmup_steps: int) -> None:
        if warmup_steps <= 0:
            self.anchor_admission_gate_schedule_progress = 1.0
        else:
            self.anchor_admission_gate_schedule_progress = min(max(float(current_step) / float(warmup_steps), 0.0), 1.0)

    def anchor_coordinate_input_effective_strength(self) -> float:
        return self.anchor_coordinate_input_strength * self.anchor_coordinate_input_schedule_progress

    def _input_missing_mask(self, history_data: torch.Tensor) -> torch.Tensor:
        target_values = history_data[:, :, :, :1]
        if self.input_missing_mask_policy == "none":
            return torch.zeros_like(target_values, dtype=torch.bool)
        if self.input_missing_mask_policy == "nan_only":
            return torch.isnan(target_values)
        if self.input_missing_mask_policy == "zero_as_missing":
            return (target_values - self.missing_zero_value).abs() <= self.missing_mask_tolerance
        raise ValueError(f"Unsupported input_missing_mask_policy: {self.input_missing_mask_policy}")

    @staticmethod
    def _masked_history_mean(values: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if valid_mask is None:
            return values.mean(dim=1)
        weights = valid_mask.to(dtype=values.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / denom

    @staticmethod
    def _masked_history_abs_mean(values: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if valid_mask is None:
            return values.abs().mean(dim=1)
        weights = valid_mask.to(dtype=values.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (values.abs() * weights).sum(dim=1) / denom

    @staticmethod
    def _recent_volatility(traffic: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if traffic.shape[1] <= 1:
            return torch.zeros(traffic.shape[0], traffic.shape[2], device=traffic.device, dtype=traffic.dtype)
        diffs = traffic[:, 1:, :] - traffic[:, :-1, :]
        diffs = diffs[:, -min(3, diffs.shape[1]) :, :]
        if valid_mask is None:
            return diffs.abs().mean(dim=1)
        valid_pairs = (valid_mask[:, 1:, :] & valid_mask[:, :-1, :])[:, -diffs.shape[1] :, :]
        weights = valid_pairs.to(dtype=traffic.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (diffs.abs() * weights).sum(dim=1) / denom

    @staticmethod
    def _tail(values: torch.Tensor, length: int) -> torch.Tensor:
        return values[:, -min(int(length), values.shape[1]) :, :]

    def _anchor_residual_statistics(
        self,
        anchor_residual: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual_for_gate = anchor_residual.detach()
        scale = self._masked_history_abs_mean(residual_for_gate, valid_mask).detach().clamp_min(1e-3)
        normalized = residual_for_gate / scale.unsqueeze(1)
        if valid_mask is not None:
            normalized = torch.where(valid_mask, normalized, torch.zeros_like(normalized))

        tail3 = self._tail(normalized, 3)
        tail6 = self._tail(normalized, 6)
        mean3 = tail3.mean(dim=1)
        mean6 = tail6.mean(dim=1)
        mean_all = self._masked_history_mean(normalized, valid_mask)
        abs3 = tail3.abs().mean(dim=1)
        abs6 = tail6.abs().mean(dim=1)
        std6 = tail6.std(dim=1, unbiased=False)
        if normalized.shape[1] >= 3:
            slope = normalized[:, -1, :] - normalized[:, -3, :]
            accel = normalized[:, -1, :] - 2.0 * normalized[:, -2, :] + normalized[:, -3, :]
        elif normalized.shape[1] >= 2:
            slope = normalized[:, -1, :] - normalized[:, -2, :]
            accel = torch.zeros_like(slope)
        else:
            slope = torch.zeros_like(normalized[:, -1, :])
            accel = torch.zeros_like(slope)
        sign_consistency = torch.sign(tail6).mean(dim=1).abs()
        if valid_mask is None:
            missing_ratio = torch.zeros_like(mean3)
        else:
            missing_ratio = 1.0 - valid_mask.to(dtype=normalized.dtype).mean(dim=1)
        return torch.stack(
            (
                mean3,
                mean6,
                mean_all,
                abs3,
                abs6,
                std6,
                slope,
                accel,
                sign_consistency,
                missing_ratio,
            ),
            dim=-1,
        )

    def _anchor_admission_gate_features(
        self,
        history_tod: torch.Tensor,
        history_dow: torch.Tensor,
        traffic: torch.Tensor,
        anchor_history_reference: torch.Tensor,
        anchor_residual: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _, num_nodes = traffic.shape
        anchor_energy = self._masked_history_abs_mean(anchor_history_reference, valid_mask)
        residual_energy = self._masked_history_abs_mean(anchor_residual, valid_mask)
        residual_anchor_ratio = (residual_energy / anchor_energy.clamp_min(1e-6)).clamp(0.0, 10.0)
        recent_volatility = self._recent_volatility(traffic, valid_mask)
        residual_stats = self._anchor_residual_statistics(anchor_residual, valid_mask)
        tod_emb = self.time_in_day_embedding(history_tod[:, -1, :]).detach()
        dow_emb = self.day_in_week_embedding(history_dow[:, -1, :]).detach()
        node_ids = torch.arange(num_nodes, device=traffic.device)
        node_context = self.anchor_admission_node_embedding(node_ids).to(dtype=traffic.dtype)
        node_context = node_context.unsqueeze(0).expand(batch_size, -1, -1)
        scalar_features = torch.stack([anchor_energy, residual_energy, residual_anchor_ratio, recent_volatility], dim=-1)
        return torch.cat([node_context, tod_emb, dow_emb, residual_stats, scalar_features.detach()], dim=-1), {
            "anchor_energy": anchor_energy,
            "residual_energy": residual_energy,
            "residual_anchor_ratio": residual_anchor_ratio,
            "recent_volatility": recent_volatility,
            "anchor_residual_stats": residual_stats,
            "anchor_residual_scale": self._masked_history_abs_mean(anchor_residual.detach(), valid_mask),
        }

    def _construct_anchor_coordinate_inputs(
        self,
        history_data: torch.Tensor,
        history_tod: torch.Tensor,
        history_dow: torch.Tensor,
        reference_daily_history: torch.Tensor,
        reference_weekly_history: torch.Tensor,
        reference_daily_future: torch.Tensor,
        reference_weekly_future: torch.Tensor,
    ) -> AnchorCoordinateInputs:
        traffic = history_data[:, :, :, 0]
        A_history = reference_daily_history + reference_weekly_history
        anchor_residual = traffic - A_history
        anchor_residual_for_input = (
            anchor_residual.detach() if self.anchor_residual_input_detach else anchor_residual
        )
        valid_mask = None
        if self.use_input_missing_mask:
            valid_mask = ~self._input_missing_mask(history_data).squeeze(-1)
        if self.anchor_admission_gate is not None:
            gate_features, gate_stats = self._anchor_admission_gate_features(
                history_tod,
                history_dow,
                traffic,
                A_history,
                anchor_residual,
                valid_mask,
            )
            gate_ratio = self.anchor_admission_gate(gate_features)
            progress = float(self.anchor_admission_gate_schedule_progress)
            fixed_ratio = anchor_residual.new_full(gate_ratio.shape, self.anchor_coordinate_input_anchor_residual_leak)
            admission_ratio = (1.0 - progress) * fixed_ratio + progress * gate_ratio
            gate_status = "enabled_nodewise_anchor_admission"
        else:
            fallback_residual_stats = self._anchor_residual_statistics(anchor_residual, valid_mask)
            gate_stats = {
                "anchor_energy": A_history.abs().mean(dim=1),
                "residual_energy": anchor_residual.abs().mean(dim=1),
                "residual_anchor_ratio": anchor_residual.abs().mean(dim=1) / A_history.abs().mean(dim=1).clamp_min(1e-6),
                "recent_volatility": self._recent_volatility(traffic, valid_mask),
                "anchor_residual_stats": fallback_residual_stats,
                "anchor_residual_scale": self._masked_history_abs_mean(anchor_residual.detach(), valid_mask),
            }
            admission_ratio = anchor_residual.new_full(
                (history_data.shape[0], self.num_nodes),
                self.anchor_coordinate_input_anchor_residual_leak,
            )
            gate_status = "fixed_anchor_admission_gate"
        strength = self.anchor_coordinate_input_effective_strength()
        anchor_decomposed = A_history + admission_ratio.unsqueeze(1) * anchor_residual_for_input
        anchor_signal = traffic + strength * (anchor_decomposed - traffic)
        anchor_history = history_data.clone()
        anchor_history[:, :, :, 0] = anchor_signal
        future_reference = reference_daily_future + reference_weekly_future
        residual_side_input_status = self._residual_side_input_status()
        aux: TensorAux = {
            "anchor_coordinate_input_status": "enabled_corrected_anchor_reference",
            "anchor_coordinate_input_mode": self.anchor_input_mode,
            "anchor_input_mode": self.anchor_input_mode,
            "anchor_coordinate_input_effective_strength": strength,
            "anchor_coordinate_input_schedule_progress": torch.tensor(
                self.anchor_coordinate_input_schedule_progress,
                device=history_data.device,
                dtype=history_data.dtype,
            ),
            "anchor_coordinate_A_history": A_history,
            "anchor_coordinate_residual": anchor_residual_for_input,
            "anchor_residual_side_input_enabled": self.anchor_residual_side_input_enabled,
            "anchor_residual_side_input_status": residual_side_input_status,
            "anchor_coordinate_daily_history": reference_daily_history,
            "anchor_coordinate_weekly_history": reference_weekly_history,
            "anchor_coordinate_recent_volatility": gate_stats["recent_volatility"],
            "anchor_admission_residual_stats_abs_mean": gate_stats["anchor_residual_stats"].detach().abs().mean(),
            "anchor_admission_residual_scale_mean": gate_stats["anchor_residual_scale"].detach().mean(),
            "anchor_admission_gate_status": gate_status,
            "anchor_admission_gate_ratio": admission_ratio,
            "anchor_admission_gate_ratio_mean": admission_ratio.detach().mean(),
            "anchor_admission_gate_ratio_std": admission_ratio.detach().std(unbiased=False),
            "anchor_admission_gate_ratio_min": admission_ratio.detach().min(),
            "anchor_admission_gate_ratio_max": admission_ratio.detach().max(),
            "anchor_admission_gate_schedule_progress": torch.tensor(
                self.anchor_admission_gate_schedule_progress,
                device=history_data.device,
                dtype=history_data.dtype,
            ),
            "corrected_anchor_daily_future": reference_daily_future,
            "corrected_anchor_weekly_future": reference_weekly_future,
            "corrected_anchor_reference_future": future_reference,
        }
        return AnchorCoordinateInputs(
            anchor_history=anchor_history,
            anchor_residual=anchor_residual_for_input,
            aux=aux,
        )

    def _prepare_inputs(
        self,
        history_data: torch.Tensor,
        history_tod: torch.Tensor,
        history_dow: torch.Tensor,
        anchor_residual_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        traffic = history_data[:, :, :, : self.num_feat]
        if self.use_input_missing_mask:
            missing_mask = self._input_missing_mask(history_data)
            traffic = torch.where(
                missing_mask.expand_as(traffic),
                torch.full_like(traffic, self.input_missing_fill_value),
                traffic,
            )
            traffic = torch.cat([traffic, missing_mask.to(traffic.dtype)], dim=-1)
        if self.anchor_residual_side_input_enabled:
            residual_input = anchor_residual_input.unsqueeze(-1)
            if self.anchor_residual_input_detach:
                residual_input = residual_input.detach()
            traffic = torch.cat([traffic, residual_input], dim=-1)
        return traffic, self.time_in_day_embedding(history_tod), self.day_in_week_embedding(history_dow)

    def encode_temporal_state(
        self,
        anchor_history: torch.Tensor,
        history_tod: torch.Tensor,
        history_dow: torch.Tensor,
        *,
        anchor_residual_input: torch.Tensor,
    ) -> torch.Tensor:
        traffic, time_in_day, day_in_week = self._prepare_inputs(
            anchor_history,
            history_tod,
            history_dow,
            anchor_residual_input,
        )
        return self.temporal_encoder(traffic, time_in_day, day_in_week)

    def get_anchor_support(self, return_aux: bool = False):
        anchor_support = self.physical_adj
        if not return_aux:
            return anchor_support
        return anchor_support, {
            "anchor_support_is_dynamic": False,
            "anchor_support_mode": "fixed_physical",
            "structure_propagation_mode": "physical_st_only",
        }

    def forward(
        self,
        history_data: torch.Tensor,
        *,
        history_tod: torch.Tensor,
        history_dow: torch.Tensor,
        reference_daily_history: torch.Tensor,
        reference_weekly_history: torch.Tensor,
        reference_daily_future: torch.Tensor,
        reference_weekly_future: torch.Tensor,
    ) -> Dict[str, object]:
        anchor_inputs = self._construct_anchor_coordinate_inputs(
            history_data,
            history_tod,
            history_dow,
            reference_daily_history,
            reference_weekly_history,
            reference_daily_future,
            reference_weekly_future,
        )

        temporal_state = self.encode_temporal_state(
            anchor_inputs.anchor_history,
            history_tod,
            history_dow,
            anchor_residual_input=anchor_inputs.anchor_residual,
        )
        anchor_support, support_aux = self.get_anchor_support(return_aux=True)

        h_t_pre_alternating = temporal_state
        temporal_state, alternating_aux = self.alternating_st_encoder(temporal_state, anchor_support)
        alternating_aux.update(
            {
                "H_t_pre_alternating": h_t_pre_alternating,
                "H_t_refined_unmixed": temporal_state,
            }
        )

        structure_sequence = temporal_state
        temporal_aux: TensorAux = {
            "temporal_encoder_type": self.temporal_encoder_type,
            "temporal_dilations": list(self.temporal_dilations) if self.temporal_dilations is not None else None,
            "temporal_layer_norm": self.temporal_layer_norm,
            "input_missing_mask_status": "enabled" if self.use_input_missing_mask else "disabled",
            "input_missing_mask_policy": self.input_missing_mask_policy if self.use_input_missing_mask else None,
            "input_missing_mask": self._input_missing_mask(history_data) if self.use_input_missing_mask else None,
            "input_missing_ratio": (
                self._input_missing_mask(history_data).float().mean() if self.use_input_missing_mask else None
            ),
            "target_zero_is_valid": self.target_zero_is_valid,
            "input_missing_fill_value": self.input_missing_fill_value if self.use_input_missing_mask else None,
            "anchor_residual_side_input_enabled": self.anchor_residual_side_input_enabled,
            "anchor_input_mode": self.anchor_input_mode,
        }
        aux: TensorAux = {
            "H_t": temporal_state,
            "residual_support_graph": anchor_support,
            "H_struct_seq": structure_sequence,
            "h_struct": structure_sequence[:, -1, :, :],
            "prior_adj": self.prior_adj,
            "physical_adj": self.physical_adj,
            "structure_encoder_status": "physical_st_only",
            **support_aux,
            **anchor_inputs.aux,
            **temporal_aux,
            **alternating_aux,
        }
        return {
            "temporal_state": temporal_state,
            "structure_sequence": structure_sequence,
            "anchor_residual": anchor_inputs.anchor_residual,
            "aux": aux,
        }
