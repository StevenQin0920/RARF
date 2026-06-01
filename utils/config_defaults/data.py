from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

DATA_DEFAULTS: Dict[str, Any] = {
    "dataset_root": "datasets",
    "data_artifact_mode": "split_npz",
    "num_feat": 1,
    "input_feature_dim": 5,
    "input_missing_fill_value": 0.0,
    "missing_mask_tolerance": 1e-6,
}

PATH_OVERRIDE_KEYS = {
    "dataset_root",
    "prior_graph_path",
    "physical_graph_path",
    "regime_anchor_field_daily_path",
    "regime_anchor_field_weekly_path",
}


def dataset_path(dataset_root: str, dataset: str, *parts: str) -> str:
    return "/".join([dataset_root.rstrip("/\\"), dataset, *parts])


def resolve_data_config(source_data: Dict[str, Any]) -> Dict[str, Any]:
    data = deepcopy(DATA_DEFAULTS)
    dataset = str(source_data["dataset"])
    target_value_type = str(source_data["target_value_type"]).lower()
    paths = source_data.get("paths", {})
    dataset_root = str(paths.get("dataset_root", data["dataset_root"]))
    data.update(
        {
            "dataset": dataset,
            "num_nodes": int(source_data["num_nodes"]),
            "target_value_type": target_value_type,
            "dataset_root": dataset_root,
            "input_missing_mask_policy": "none" if target_value_type == "flow" else "zero_as_missing",
            "prior_graph_path": dataset_path(dataset_root, dataset, "graphs", "A_0.pkl"),
            "physical_graph_path": dataset_path(dataset_root, dataset, "graphs", "A_phy.pkl"),
            "regime_anchor_field_daily_path": dataset_path(
                dataset_root,
                dataset,
                "anchors",
                "regime_anchor_field_daily.npy",
            ),
            "regime_anchor_field_weekly_path": dataset_path(
                dataset_root,
                dataset,
                "anchors",
                "regime_anchor_field_weekly.npy",
            ),
        }
    )
    for key in PATH_OVERRIDE_KEYS:
        if key in paths:
            data[key] = str(paths[key])
    return data
