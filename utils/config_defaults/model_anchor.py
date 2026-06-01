from __future__ import annotations

from typing import Any, Dict


MODEL_CORE_DEFAULTS: Dict[str, Any] = {
    "model_variant": "rarf",
    "graph_profile": "rarf_physical_residual",
    "anchor_support_mode": "fixed_physical",
    "hidden_dim": 64,
    "node_dim": 32,
    "time_emb_dim": 16,
    "output_hidden_dim": 256,
    "horizon": 12,
    "history_length": 12,
    "tod_vocab_size": 288,
    "dow_vocab_size": 7,
    "weekend_vocab_size": 2,
    "holiday_vocab_size": 2,
    "future_context_tod_channel_index": 1,
    "future_context_dow_channel_index": 2,
    "future_context_weekend_channel_index": 3,
    "future_context_holiday_channel_index": 4,
    "weekend_channel_index": 3,
    "holiday_channel_index": 4,
    "holiday_mode": "input_channel",
    "dropout": 0.1,
    "temporal_dropout": 0.1,
    "graph_dropout": 0.1,
    "residual_encoder_dropout": 0.1,
    "residual_decoder_dropout": 0.1,
    "use_input_missing_mask": True,
}

REGIME_ANCHOR_DEFAULTS: Dict[str, Any] = {
    "regime_daily_weight": 1.0,
    "regime_weekly_weight": 1.0,
}

TEMPORAL_BIAS_ENCODER_DEFAULTS: Dict[str, Any] = {
    "temporal_encoder_type": "gated_tcn",
    "temporal_layers": 3,
    "temporal_dilations": [1, 2, 4],
    "temporal_layer_norm": True,
    "temporal_kernel_size": 3,
    "use_alternating_st_encoder": True,
    "alternating_st_layers": 2,
    "alternating_st_kernel_size": 3,
    "alternating_st_dilations": [1, 2],
    "alternating_st_temporal_dropout": 0.1,
    "alternating_st_spatial_dropout": 0.12,
    "alternating_st_supports": ["identity", "forward", "backward"],
    "alternating_st_gate": True,
    "alternating_st_layer_norm": True,
}

ANCHOR_INPUT_DEFAULTS: Dict[str, Any] = {
    "anchor_input_mode": "learned_alpha_with_residual_side_input",
    "use_anchor_admission_gate": True,
    "anchor_residual_input_detach": True,
    "anchor_coordinate_input_anchor_residual_leak": 0.2,
    "anchor_coordinate_input_strength": 1.0,
    "anchor_admission_gate_ratio_min": 0.0,
    "anchor_admission_gate_ratio_max": 0.8,
    "anchor_admission_gate_ratio_init": 0.2,
    "anchor_admission_node_emb_dim": 16,
    "anchor_admission_gate_hidden_dim": 64,
}

TEMPORAL_BIAS_BRANCH_DEFAULTS: Dict[str, Any] = {
    "horizon_emb_dim": 16,
    "horizon_readout_dropout": 0.1,
    "use_horizon_forecast_decoder": True,
    "horizon_decoder_layers": 2,
    "horizon_decoder_kernel_size": 3,
    "horizon_decoder_dilations": [1, 2],
    "horizon_decoder_dropout": 0.12,
    "horizon_decoder_layer_norm": True,
}
