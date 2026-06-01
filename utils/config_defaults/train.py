from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

TRAIN_DEFAULTS: Dict[str, Any] = {
    "seed": 1,
    "deterministic": True,
    "deterministic_algorithms": False,
    "deterministic_warn_only": True,
    "cudnn_benchmark": False,
    "torch_num_threads": None,
    "device": "cuda",
    "epochs": 100,
    "batch_size": 32,
    "valid_batch_size": 32,
    "test_batch_size": 32,
    "learning_rate": 0.001,
    "fft_loss_weight": 0.0,
    "optimizer": "adamw",
    "weight_decay": 0.0001,
    "no_decay_weight_decay": 0.0,
    "scheduler": "warmup_cosine",
    "warmup_steps": 0,
    "warmup_epochs": 2.0,
    "min_lr_ratio": 0.05,
    "use_grad_clip": True,
    "grad_clip_norm": 5.0,
    "use_amp": True,
    "use_ema": True,
    "ema_decay": 0.999,
    "ema_eval": True,
    "early_stopping": True,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.0001,
    "early_stopping_min_epochs": 25,
    "early_stopping_mode": "min",
    "anchor_coordinate_input_warmup_epochs": 2.0,
    "anchor_admission_gate_warmup_epochs": 5.0,
}

EVAL_DEFAULTS: Dict[str, Any] = {
    "report_horizons": [3, 6, 12],
    "offload_predictions_to_cpu": True,
}

OUTPUT_DEFAULTS: Dict[str, Any] = {"root": "output"}


def resolve_train_config(source_train: Dict[str, Any]) -> Dict[str, Any]:
    train = deepcopy(TRAIN_DEFAULTS)
    train.update(source_train)
    batch_size = int(train["batch_size"])
    train["valid_batch_size"] = batch_size
    train["test_batch_size"] = batch_size
    return train
