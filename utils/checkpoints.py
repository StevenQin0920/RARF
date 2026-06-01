from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

def clone_state_dict_to_cpu(state_dict: Dict) -> Dict:
    cloned = {}
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[name] = value.detach().cpu().clone()
        else:
            cloned[name] = value
    return cloned

class ModelEMA:
    """Optional exponential moving average for model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }
        self.backup = {}

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = {}
        state = model.state_dict()
        for name, shadow_value in self.shadow.items():
            self.backup[name] = state[name].detach().cpu().clone()
            state[name].copy_(shadow_value)

    def restore(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.backup.items():
            state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))
        self.backup = {}

    def state_dict(self) -> Dict:
        return {
            "decay": self.decay,
            "shadow": clone_state_dict_to_cpu(self.shadow),
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.shadow = {
            name: value.detach().clone()
            for name, value in state_dict["shadow"].items()
            if torch.is_tensor(value)
        }

def overlay_state_dict(model: torch.nn.Module, overlay: Dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    missing = [name for name in overlay if name not in state]
    if missing:
        raise KeyError(f"EMA shadow contains keys not present in model state_dict: {missing[:5]}")
    for name, value in overlay.items():
        state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))

def save_best_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: Optional[ModelEMA],
    epoch: int,
    val_loss: float,
    eval_uses_ema: bool,
    global_step: int,
) -> Dict:
    best_checkpoint_type = "ema_shadow" if eval_uses_ema else "raw_model"
    checkpoint = {
        "checkpoint_format": "rarf_best_checkpoint",
        "raw_model_state_dict": clone_state_dict_to_cpu(model.state_dict()),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "best_checkpoint_type": best_checkpoint_type,
        "best_epoch": int(epoch),
        "best_val_mae": float(val_loss),
        "global_step": int(global_step),
        "ema_enabled": ema is not None,
        "ema_eval": bool(eval_uses_ema),
        "best_checkpoint_includes_ema_state": ema is not None,
        "test_weight_source": best_checkpoint_type,
    }
    torch.save(checkpoint, path)
    return checkpoint

def load_best_checkpoint_for_test(model: torch.nn.Module, path: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_format") == "rarf_best_checkpoint":
        model.load_state_dict(checkpoint["raw_model_state_dict"])
        test_weight_source = checkpoint.get("test_weight_source", checkpoint.get("best_checkpoint_type", "raw_model"))
        if test_weight_source == "ema_shadow":
            ema_state = checkpoint.get("ema_state_dict")
            if ema_state is None or "shadow" not in ema_state:
                raise ValueError("Best checkpoint expects EMA shadow for test, but ema_state_dict is missing.")
            overlay_state_dict(model, ema_state["shadow"])
        elif test_weight_source != "raw_model":
            raise ValueError(f"Unsupported test_weight_source `{test_weight_source}`.")
        return {
            "checkpoint_format": checkpoint.get("checkpoint_format"),
            "best_checkpoint_type": checkpoint.get("best_checkpoint_type", test_weight_source),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_val_mae": checkpoint.get("best_val_mae"),
            "ema_enabled": bool(checkpoint.get("ema_enabled", False)),
            "ema_eval": bool(checkpoint.get("ema_eval", False)),
            "best_checkpoint_includes_ema_state": bool(checkpoint.get("best_checkpoint_includes_ema_state", False)),
            "test_weight_source": test_weight_source,
            "test_weights_aligned_with_best_validation": True,
        }

    model.load_state_dict(checkpoint)
    return {
        "checkpoint_format": "raw_state_dict_checkpoint",
        "best_checkpoint_type": "raw_model",
        "best_epoch": None,
        "best_val_mae": None,
        "ema_enabled": False,
        "ema_eval": False,
        "best_checkpoint_includes_ema_state": False,
        "test_weight_source": "raw_model",
        "test_weights_aligned_with_best_validation": "unknown_raw_state_dict_checkpoint",
    }

def load_checkpoint_into_model(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    weight_source: str = "checkpoint_default",
    strict: bool = True,
) -> Dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_format") == "rarf_best_checkpoint":
        model.load_state_dict(checkpoint["raw_model_state_dict"], strict=strict)
        resolved_source = weight_source
        if resolved_source == "checkpoint_default":
            resolved_source = checkpoint.get("test_weight_source", checkpoint.get("best_checkpoint_type", "raw_model"))
        if resolved_source == "ema_shadow":
            ema_state = checkpoint.get("ema_state_dict")
            if ema_state is None or "shadow" not in ema_state:
                raise ValueError("Requested EMA shadow initialization, but ema_state_dict is missing.")
            overlay_state_dict(model, ema_state["shadow"])
        elif resolved_source != "raw_model":
            raise ValueError(f"Unsupported initial checkpoint weight_source `{resolved_source}`.")
        return {
            "checkpoint_format": checkpoint.get("checkpoint_format"),
            "weight_source": resolved_source,
            "best_epoch": checkpoint.get("best_epoch"),
            "best_val_mae": checkpoint.get("best_val_mae"),
            "ema_enabled": bool(checkpoint.get("ema_enabled", False)),
            "path": str(path),
        }

    model.load_state_dict(checkpoint, strict=strict)
    return {
        "checkpoint_format": "raw_state_dict_checkpoint",
        "weight_source": "raw_model",
        "best_epoch": None,
        "best_val_mae": None,
        "ema_enabled": False,
        "path": str(path),
    }


