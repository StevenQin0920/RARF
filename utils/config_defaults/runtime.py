from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .data import resolve_data_config
from .method import METHOD_DEFAULTS
from .model_anchor import (
    ANCHOR_INPUT_DEFAULTS,
    MODEL_CORE_DEFAULTS,
    REGIME_ANCHOR_DEFAULTS,
    TEMPORAL_BIAS_BRANCH_DEFAULTS,
    TEMPORAL_BIAS_ENCODER_DEFAULTS,
)
from .model_regime import FUTURE_CONTEXT_DEFAULTS
from .train import EVAL_DEFAULTS, OUTPUT_DEFAULTS, resolve_train_config

_HIDDEN_DIM_FOLLOWERS = (
    "anchor_admission_gate_hidden_dim",
    "future_context_hidden_dim",
)

_DROPOUT_FOLLOWERS = (
    "temporal_dropout",
    "graph_dropout",
    "residual_encoder_dropout",
    "residual_decoder_dropout",
)


def build_model_defaults() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for group in (
        MODEL_CORE_DEFAULTS,
        REGIME_ANCHOR_DEFAULTS,
        TEMPORAL_BIAS_ENCODER_DEFAULTS,
        ANCHOR_INPUT_DEFAULTS,
        FUTURE_CONTEXT_DEFAULTS,
        TEMPORAL_BIAS_BRANCH_DEFAULTS,
    ):
        defaults.update(deepcopy(group))
    return defaults


def resolve_model_config(source_model: Dict[str, Any]) -> Dict[str, Any]:
    model = build_model_defaults()
    model.update(source_model)
    if "hidden_dim" in source_model:
        hidden_dim = int(source_model["hidden_dim"])
        for key in _HIDDEN_DIM_FOLLOWERS:
            model[key] = hidden_dim
    if "dropout" in source_model:
        dropout = float(source_model["dropout"])
        for key in _DROPOUT_FOLLOWERS:
            model[key] = dropout
    return model


def resolve_runtime_config(source_config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a slim public config into the full RARF runtime configuration."""
    return {
        "method": deepcopy(METHOD_DEFAULTS),
        "data": resolve_data_config(source_config["data"]),
        "model": resolve_model_config(source_config.get("model", {})),
        "train": resolve_train_config(source_config["train"]),
        "eval": {**deepcopy(EVAL_DEFAULTS), **source_config.get("eval", {})},
        "output": {**deepcopy(OUTPUT_DEFAULTS), **source_config.get("output", {})},
    }
