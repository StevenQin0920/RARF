from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from utils.io import iter_batches
from utils.losses import MaskValue, fft_magnitude_loss, mae_loss, valid_value_mask


def run_training_epoch(
    trainer,
    *,
    max_batches: Optional[int],
    train_mask_value: MaskValue,
    method_schedule_steps: dict[str, int],
) -> tuple[float, Dict]:
    """Run one RARF training epoch and return mean losses."""
    trainer.model.train()
    total_losses: list[float] = []
    mae_losses: list[float] = []
    fft_losses: list[float] = []
    fft_loss_weight = float(trainer.train_config.get("fft_loss_weight", 0.0))
    counters = {
        "optimizer_steps": 0,
        "skipped_optimizer_steps": 0,
        "nonfinite_label_batches": 0,
        "nonfinite_prediction_batches": 0,
        "nonfinite_loss_batches": 0,
    }

    for _, x, y in iter_batches(trainer.train_loader, trainer.device, max_batches):
        trainer.optimizer.zero_grad(set_to_none=True)
        labels = y[:, :, :, 0].transpose(1, 2)
        labels_inv = trainer.scaler.inverse_transform(labels)
        if has_nonfinite_valid_values(labels_inv, labels_inv, train_mask_value):
            _skip_step(trainer, counters, "nonfinite_label_batches")
            continue

        _update_method_schedules(trainer, method_schedule_steps)
        with _autocast(trainer):
            preds = trainer.model(x, future_data=y)
            preds_inv = trainer.scaler.inverse_transform(preds)
            loss, loss_parts = compute_training_loss(
                preds_inv,
                labels_inv,
                train_mask_value=train_mask_value,
                fft_loss_weight=fft_loss_weight,
            )

        if has_nonfinite_valid_values(preds_inv.detach(), labels_inv, train_mask_value):
            _skip_step(trainer, counters, "nonfinite_prediction_batches")
            continue
        if not torch.isfinite(loss):
            _skip_step(trainer, counters, "nonfinite_loss_batches")
            continue

        _backward(trainer, loss)
        _clip_gradients(trainer)
        if _optimizer_step(trainer):
            counters["optimizer_steps"] += 1
        else:
            counters["skipped_optimizer_steps"] += 1

        total_loss_value = float(loss.detach().item())
        mae_loss_value = float(loss_parts["mae_loss"].detach().item())
        total_losses.append(total_loss_value)
        mae_losses.append(mae_loss_value)
        if loss_parts["fft_loss"] is not None:
            fft_losses.append(float(loss_parts["fft_loss"].detach().item()))
        trainer.global_step += 1
        _update_method_schedules(trainer, method_schedule_steps)

    return summarize_training_epoch(
        total_losses,
        mae_losses,
        fft_losses,
        counters,
        fft_loss_weight,
    )


def _update_method_schedules(trainer, method_schedule_steps: dict[str, int]) -> None:
    current_step = int(trainer.global_step)
    if hasattr(trainer.model, "set_anchor_coordinate_input_schedule"):
        trainer.model.set_anchor_coordinate_input_schedule(
            current_step,
            int(method_schedule_steps.get("anchor_coordinate_input", 0)),
        )
    if hasattr(trainer.model, "set_anchor_admission_gate_schedule"):
        trainer.model.set_anchor_admission_gate_schedule(
            current_step,
            int(method_schedule_steps.get("anchor_admission_gate", 0)),
        )
def compute_training_loss(
    preds_inv: torch.Tensor,
    labels_inv: torch.Tensor,
    *,
    train_mask_value: MaskValue,
    fft_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    mae = mae_loss(preds_inv, labels_inv, train_mask_value)
    fft_loss = None
    total_loss = mae
    if float(fft_loss_weight) > 0.0:
        fft_loss = fft_magnitude_loss(preds_inv, labels_inv, train_mask_value)
        total_loss = mae + float(fft_loss_weight) * fft_loss
    return total_loss, {
        "mae_loss": mae,
        "fft_loss": fft_loss,
    }


def has_nonfinite_valid_values(values: torch.Tensor, labels: torch.Tensor, mask_value: MaskValue) -> bool:
    valid = valid_value_mask(labels, mask_value)
    if not bool(valid.any().item()):
        return True
    return bool((~torch.isfinite(values) & valid).any().item())


def _skip_step(trainer, counters: dict[str, int], reason: str) -> None:
    counters[reason] += 1
    counters["skipped_optimizer_steps"] += 1
    trainer.global_step += 1


def _autocast(trainer):
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=trainer.amp_enabled)
    return torch.cuda.amp.autocast(enabled=trainer.amp_enabled)


def _backward(trainer, loss: torch.Tensor) -> None:
    if trainer.amp_scaler is not None and trainer.amp_enabled:
        trainer.amp_scaler.scale(loss).backward()
        trainer.amp_scaler.unscale_(trainer.optimizer)
    else:
        loss.backward()


def _clip_gradients(trainer) -> None:
    if not bool(trainer.train_config.get("use_grad_clip", True)):
        return
    grad_clip_norm = float(trainer.train_config.get("grad_clip_norm", 5.0))
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), grad_clip_norm)


def _optimizer_step(trainer) -> bool:
    step_performed = True
    if trainer.amp_scaler is not None and trainer.amp_enabled:
        scale_before = trainer.amp_scaler.get_scale()
        trainer.amp_scaler.step(trainer.optimizer)
        trainer.amp_scaler.update()
        step_performed = trainer.amp_scaler.get_scale() >= scale_before
    else:
        trainer.optimizer.step()
    if step_performed:
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        if trainer.ema is not None:
            trainer.ema.update(trainer.model)
    return step_performed


def summarize_training_epoch(
    total_losses: list[float],
    mae_losses: list[float],
    fft_losses: list[float],
    counters: dict[str, int],
    fft_loss_weight: float,
) -> tuple[float, Dict]:
    mean_total_loss = float(np.mean(total_losses)) if total_losses else float("nan")
    mean_mae_loss = float(np.mean(mae_losses)) if mae_losses else float("nan")
    mean_fft_loss = float(np.mean(fft_losses)) if fft_losses else None
    stats = {
        "train_total_loss": mean_total_loss,
        "train_mae_loss": mean_mae_loss,
        "train_fft_loss": mean_fft_loss,
        "fft_loss_weight": float(fft_loss_weight),
    }
    if counters["skipped_optimizer_steps"] > 0:
        stats["skipped_batches"] = int(counters["skipped_optimizer_steps"])
    return mean_mae_loss, stats
