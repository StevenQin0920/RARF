from __future__ import annotations

import math
from typing import Dict

import torch


def build_parameter_groups(model: torch.nn.Module, config: Dict):
    weight_decay = float(config.get("weight_decay", 0.0))
    no_decay_weight_decay = float(config.get("no_decay_weight_decay", 0.0))

    no_decay_lr_multiplier = float(config.get("no_decay_lr_multiplier", 1.0))

    decay_params = []
    no_decay_params = []
    group_names = {
        "decay": [],
        "no_decay": [],
    }

    def is_no_decay_parameter(param_name: str, param) -> bool:
        lowered_name = param_name.lower()
        return param.ndim <= 1 or param_name.endswith(".bias") or "norm" in lowered_name or "embedding" in lowered_name

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_no_decay_parameter(name, param):
            no_decay_params.append(param)
            group_names["no_decay"].append(name)
        else:
            decay_params.append(param)
            group_names["decay"].append(name)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay, "group_name": "decay"})
    if no_decay_params:
        groups.append(
            {
                "params": no_decay_params,
                "weight_decay": no_decay_weight_decay,
                "group_name": "no_decay",
                "lr_multiplier": no_decay_lr_multiplier,
            }
        )
    return groups, group_names


def _apply_group_learning_rates(parameter_groups: list[Dict], lr: float) -> None:
    for group in parameter_groups:
        multiplier = float(group.pop("lr_multiplier", 1.0))
        group["lr"] = lr * multiplier


def build_optimizer(model: torch.nn.Module, config: Dict):
    parameter_groups, group_names = build_parameter_groups(model, config)
    optimizer_name = config.get("optimizer", "adamw").lower()
    lr = float(config.get("learning_rate", 0.001))
    _apply_group_learning_rates(parameter_groups, lr)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(parameter_groups)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameter_groups)
    else:
        raise ValueError(f"Unsupported optimizer `{optimizer_name}`.")
    return optimizer, group_names


def resolve_warmup_steps(config: Dict, steps_per_epoch: int) -> int:
    if "warmup_epochs" in config:
        return max(int(round(float(config["warmup_epochs"]) * max(int(steps_per_epoch), 1))), 0)
    return max(int(config.get("warmup_steps", 0)), 0)


def build_scheduler(optimizer, config: Dict, total_steps: int, warmup_steps: int):
    scheduler_name = config.get("scheduler", "warmup_cosine").lower()
    if scheduler_name in {"none", "disabled"}:
        return None
    if scheduler_name != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler `{scheduler_name}`.")

    min_lr_ratio = float(config.get("min_lr_ratio", 0.05))
    total_steps = max(int(total_steps), 1)
    warmup_steps = min(max(int(warmup_steps), 0), total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denom = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Monitor validation metrics and stop when improvement stalls."""

    def __init__(
        self,
        enabled: bool = True,
        patience: int = 10,
        min_delta: float = 0.0,
        min_epochs: int = 0,
        mode: str = "min",
    ):
        if mode not in {"min", "max"}:
            raise ValueError(f"early_stopping_mode must be 'min' or 'max', got `{mode}`.")
        self.enabled = bool(enabled)
        self.patience = max(int(patience), 0)
        self.min_delta = float(min_delta)
        self.min_epochs = max(int(min_epochs), 0)
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.best_epoch = 0
        self.num_bad_epochs = 0
        self.should_stop = False
        self.stopped_epoch = None

    def is_better(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, epoch: int) -> Dict:
        improved = self.is_better(value)
        if improved:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.enabled and epoch >= self.min_epochs and self.num_bad_epochs >= self.patience:
            self.should_stop = True
            self.stopped_epoch = int(epoch)

        return {
            "improved": improved,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "num_bad_epochs": self.num_bad_epochs,
            "should_stop": self.should_stop,
            "stopped_epoch": self.stopped_epoch,
        }

    def state_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "mode": self.mode,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "num_bad_epochs": self.num_bad_epochs,
            "should_stop": self.should_stop,
            "stopped_epoch": self.stopped_epoch,
        }


def build_early_stopping(config: Dict) -> EarlyStopping:
    return EarlyStopping(
        enabled=bool(config.get("early_stopping", True)),
        patience=int(config.get("early_stopping_patience", 10)),
        min_delta=float(config.get("early_stopping_min_delta", 0.0)),
        min_epochs=int(config.get("early_stopping_min_epochs", 0)),
        mode=config.get("early_stopping_mode", "min"),
    )


def get_learning_rates(optimizer) -> Dict[str, float]:
    lrs = {}
    for idx, group in enumerate(optimizer.param_groups):
        group_name = group.get("group_name", f"group_{idx}")
        lrs[group_name] = float(group["lr"])
    return lrs
