from __future__ import annotations

from typing import Any, Dict

TOP_LEVEL_KEYS = {"data", "model", "train", "eval", "output"}
REQUIRED_TOP_LEVEL_KEYS = {"data", "model", "train"}

DATA_KEYS = {"dataset", "num_nodes", "target_value_type", "paths"}
DATA_PATH_KEYS = {
    "dataset_root",
    "prior_graph_path",
    "physical_graph_path",
    "regime_anchor_field_daily_path",
    "regime_anchor_field_weekly_path",
}

MODEL_KEYS = {
    "hidden_dim",
    "node_dim",
    "output_hidden_dim",
    "horizon",
    "history_length",
    "dropout",
}

TRAIN_KEYS = {
    "seed",
    "deterministic",
    "deterministic_algorithms",
    "deterministic_warn_only",
    "cudnn_benchmark",
    "torch_num_threads",
    "device",
    "epochs",
    "batch_size",
    "learning_rate",
    "fft_loss_weight",
    "optimizer",
    "weight_decay",
    "use_amp",
    "use_ema",
    "early_stopping_patience",
}

EVAL_KEYS = {"report_horizons", "offload_predictions_to_cpu"}
OUTPUT_KEYS = {"root"}

SECTION_KEYS = {
    "data": DATA_KEYS,
    "model": MODEL_KEYS,
    "train": TRAIN_KEYS,
    "eval": EVAL_KEYS,
    "output": OUTPUT_KEYS,
}


def _unknown_keys(data: Dict[str, Any], allowed: set[str]) -> list[str]:
    return sorted(key for key in data if key not in allowed)


def _validate_object(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{label}` must be an object.")
    return value


def validate_config(config: Dict[str, Any]) -> None:
    inherited_key = "_base" + "_config"
    if inherited_key in config:
        raise ValueError("RARF slim configs are self-contained; inherited config keys are not supported.")
    _validate_object(config, "root")
    top_unknown = _unknown_keys(config, TOP_LEVEL_KEYS)
    if top_unknown:
        raise ValueError(f"Unknown top-level config keys: {top_unknown}")
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    for section, allowed in SECTION_KEYS.items():
        if section not in config:
            continue
        value = _validate_object(config[section], section)
        unknown = _unknown_keys(value, allowed)
        if unknown:
            raise ValueError(f"Unknown keys in config section `{section}`: {unknown}")

    data = config["data"]
    required_data = {"dataset", "num_nodes", "target_value_type"}
    missing_data = sorted(required_data - set(data))
    if missing_data:
        raise ValueError(f"Missing required data config keys: {missing_data}")
    target_value_type = str(data.get("target_value_type", "")).lower()
    if target_value_type not in {"flow", "speed"}:
        raise ValueError("data.target_value_type must be either 'flow' or 'speed'.")
    if "paths" in data:
        paths = _validate_object(data["paths"], "data.paths")
        unknown_paths = _unknown_keys(paths, DATA_PATH_KEYS)
        if unknown_paths:
            raise ValueError(f"Unknown keys in config section `data.paths`: {unknown_paths}")

    model = config["model"]
    required_model = {"hidden_dim", "node_dim", "horizon", "history_length", "dropout"}
    missing_model = sorted(required_model - set(model))
    if missing_model:
        raise ValueError(f"Missing required model config keys: {missing_model}")

    train = config["train"]
    required_train = {"seed", "device", "epochs", "batch_size", "learning_rate"}
    missing_train = sorted(required_train - set(train))
    if missing_train:
        raise ValueError(f"Missing required train config keys: {missing_train}")
    if "optimizer" in train:
        optimizer = str(train["optimizer"]).lower()
        if optimizer not in {"adamw", "adam"}:
            raise ValueError("train.optimizer must be either 'adamw' or 'adam'.")
