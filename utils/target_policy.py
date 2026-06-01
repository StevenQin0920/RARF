from __future__ import annotations

from typing import Any, Dict


def resolve_target_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    data = config["data"]
    target_value_type = str(data["target_value_type"]).lower()
    configured_input_policy = data.get("input_missing_mask_policy")
    if target_value_type == "flow":
        input_missing_mask_policy = str(configured_input_policy or "none")
        return {
            "target_value_type": "flow",
            "metric_policy": "flow_full_mae_rmse_mape_nonzero_denominator",
            "target_zero_is_valid": True,
            "train_mask_value": None,
            "mae_mask_value": None,
            "rmse_mask_value": None,
            "mape_mask_value": None,
            # Float32 inverse-scaling can turn raw zero flow into tiny nonzero
            # values; keep MAPE's zero-denominator exclusion tolerant to that.
            "mape_eps": 1e-4,
            "input_missing_mask_policy": input_missing_mask_policy,
        }
    if target_value_type == "speed":
        input_missing_mask_policy = str(configured_input_policy or "zero_as_missing")
        return {
            "target_value_type": "speed",
            "metric_policy": "speed_zero_as_missing_mae_rmse_mape",
            "target_zero_is_valid": False,
            "train_mask_value": 0.0,
            "mae_mask_value": 0.0,
            "rmse_mask_value": 0.0,
            "mape_mask_value": 0.0,
            "mape_eps": 1e-5,
            "input_missing_mask_policy": input_missing_mask_policy,
        }
    raise ValueError("data.target_value_type must be either 'flow' or 'speed'.")
