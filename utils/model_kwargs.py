from __future__ import annotations

from typing import Any, Dict

from .config_loader import resolve_path

def build_model_kwargs(
    config: Dict[str, Any],
    *,
    prior_adj,
    physical_adj,
    target_zero_is_valid: bool,
    input_missing_mask_policy: str,
    missing_zero_value: float,
) -> Dict[str, Any]:
    data = config["data"]
    model = config["model"]
    return {
        **model,
        "num_nodes": int(data["num_nodes"]),
        "num_feat": int(data.get("num_feat", 1)),
        "num_hidden": int(model.get("hidden_dim", 64)),
        "node_hidden": int(model.get("node_dim", 32)),
        "target_zero_is_valid": target_zero_is_valid,
        "input_missing_mask_policy": input_missing_mask_policy,
        "input_missing_fill_value": float(data.get("input_missing_fill_value", 0.0)),
        "missing_zero_value": missing_zero_value,
        "missing_mask_tolerance": float(data.get("missing_mask_tolerance", 1e-5)),
        "regime_anchor_field_daily_path": str(resolve_path(data["regime_anchor_field_daily_path"])),
        "regime_anchor_field_weekly_path": str(resolve_path(data["regime_anchor_field_weekly_path"])),
        "prior_adj": prior_adj,
        "physical_adj": physical_adj,
        "spatial_bias_graph_adj": prior_adj,
        "model_variant": model.get("model_variant", "rarf"),
    }


def optimizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["train"]
