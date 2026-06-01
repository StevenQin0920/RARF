from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch

from utils.checkpoints import ModelEMA, load_best_checkpoint_for_test, save_best_checkpoint
from utils.losses import MaskValue
from utils.optim import build_early_stopping, build_optimizer, build_scheduler, get_learning_rates, resolve_warmup_steps
from utils.reporting import format_epoch_log, write_history_csv, write_rows_csv

from .evaluator import (
    evaluate_model,
    extract_horizon_metrics,
    get_average_metric,
    maybe_apply_ema,
)
from .train_loop import run_training_epoch


class RARFTrainer:
    """RARF training orchestration: epochs, validation, early stopping, and checkpoints."""

    def __init__(
        self,
        *,
        model,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        device: torch.device,
        train_config: Dict,
        eval_config: Dict,
        run_dir: Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.device = device
        self.train_config = train_config
        self.eval_config = eval_config
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.ckpt_dir / "best.pt"
        self.history_path = self.run_dir / "history.csv"
        self.test_metrics_path = self.run_dir / "test_metrics_by_horizon.csv"
        self.optimizer, self.parameter_group_names = build_optimizer(model, train_config)
        self.ema = ModelEMA(model, float(train_config.get("ema_decay", 0.999))) if bool(train_config.get("use_ema", False)) else None
        self.ema_for_eval = self.ema if bool(train_config.get("ema_eval", True)) else None
        self.amp_enabled = bool(train_config.get("use_amp", False)) and device.type == "cuda"
        self.amp_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled) if hasattr(torch, "amp") else None
        self.report_horizons = [int(horizon) for horizon in eval_config.get("report_horizons", [3, 6, 12])]
        self.global_step = 0
        self.scheduler = None
        self.early_stopping = None
        self.history: list[Dict] = []

    def train_epoch(
        self,
        *,
        max_batches: Optional[int],
        train_mask_value: MaskValue,
        method_schedule_steps: dict[str, int],
    ) -> tuple[float, Dict]:
        return run_training_epoch(
            self,
            max_batches=max_batches,
            train_mask_value=train_mask_value,
            method_schedule_steps=method_schedule_steps,
        )

    def evaluate(self, loader, **kwargs) -> list[Dict]:
        return evaluate_model(self.model, loader, self.scaler, self.device, **kwargs)

    def fit(
        self,
        *,
        epochs: int,
        max_train_batches: Optional[int],
        max_eval_batches: Optional[int],
        train_mask_value: MaskValue,
        mae_mask_value: MaskValue,
        rmse_mask_value: MaskValue,
        mape_mask_value: MaskValue,
        mape_eps: float,
    ) -> list[Dict]:
        steps_per_epoch = min(len(self.train_loader), max_train_batches) if max_train_batches is not None else len(self.train_loader)
        self._prepare_training_schedule(epochs, steps_per_epoch)
        method_schedule_steps = self._resolve_method_schedule_steps(steps_per_epoch)
        offload = bool(self.eval_config.get("offload_predictions_to_cpu", False))
        best_checkpoint = None

        for epoch in range(1, int(epochs) + 1):
            epoch_start = time.time()
            train_loss, train_stats = self.train_epoch(
                max_batches=max_train_batches,
                train_mask_value=train_mask_value,
                method_schedule_steps=method_schedule_steps,
            )
            val_rows = self._validate_with_optional_ema(
                max_eval_batches=max_eval_batches,
                mae_mask_value=mae_mask_value,
                rmse_mask_value=rmse_mask_value,
                mape_mask_value=mape_mask_value,
                mape_eps=mape_eps,
                offload=offload,
            )
            val_loss = get_average_metric(val_rows, "mae")
            stop_state = self.early_stopping.step(val_loss, epoch)
            if bool(stop_state["improved"]):
                best_checkpoint = self._save_best(epoch, val_loss)
            self._append_epoch_row(epoch, epoch_start, train_loss, train_stats, val_rows, stop_state)
            if stop_state["should_stop"]:
                break

        if best_checkpoint is None:
            self._save_fallback_checkpoint(
                max_eval_batches=max_eval_batches,
                mae_mask_value=mae_mask_value,
                rmse_mask_value=rmse_mask_value,
                mape_mask_value=mape_mask_value,
                mape_eps=mape_eps,
                offload=offload,
            )
        write_history_csv(self.history_path, self.history)
        return self.history

    def _prepare_training_schedule(self, epochs: int, steps_per_epoch: int) -> None:
        total_steps = max(int(epochs) * max(steps_per_epoch, 1), 1)
        warmup_steps = resolve_warmup_steps(self.train_config, steps_per_epoch)
        self.scheduler = build_scheduler(self.optimizer, self.train_config, total_steps, warmup_steps)
        self.early_stopping = build_early_stopping(self.train_config)

    def _resolve_method_schedule_steps(self, steps_per_epoch: int) -> dict[str, int]:
        steps_per_epoch = max(int(steps_per_epoch), 1)
        coordinate = int(
            round(float(self.train_config.get("anchor_coordinate_input_warmup_epochs", 0.0)) * steps_per_epoch)
        )
        admission = int(
            round(float(self.train_config.get("anchor_admission_gate_warmup_epochs", 0.0)) * steps_per_epoch)
        )
        return {
            "anchor_coordinate_input": max(coordinate, 0),
            "anchor_admission_gate": max(admission, 0),
        }

    def _validate_with_optional_ema(self, *, max_eval_batches: Optional[int], offload: bool, **metric_kwargs):
        return maybe_apply_ema(
            self.model,
            self.ema_for_eval,
            lambda: self.evaluate(
                self.val_loader,
                max_batches=max_eval_batches,
                offload_predictions_to_cpu=offload,
                **metric_kwargs,
            ),
        )

    def _save_best(self, epoch: int, val_loss: float):
        return save_best_checkpoint(
            self.best_path,
            self.model,
            self.ema,
            epoch,
            val_loss,
            eval_uses_ema=self.ema_for_eval is not None,
            global_step=self.global_step,
        )

    def _append_epoch_row(self, epoch: int, epoch_start: float, train_loss: float, train_stats: Dict, val_rows, stop_state):
        current_lrs = get_learning_rates(self.optimizer)
        row = {
            "epoch": epoch,
            "mode": "RARF",
            "train_mae": train_loss,
            "val_mae": get_average_metric(val_rows, "mae"),
            "val_rmse": get_average_metric(val_rows, "rmse"),
            "val_mape": get_average_metric(val_rows, "mape"),
            "best_val_mae": stop_state["best"],
            "best_epoch": stop_state["best_epoch"],
            "lr": current_lrs.get("decay", next(iter(current_lrs.values()))),
            "is_best": bool(stop_state["improved"]),
            "early_stop": stop_state["should_stop"],
            "epoch_time_sec": time.time() - epoch_start,
            **train_stats,
            **extract_horizon_metrics(val_rows, self.report_horizons, "val"),
        }
        self.history.append(row)
        print(format_epoch_log(row))

    def _save_fallback_checkpoint(self, *, max_eval_batches: Optional[int], offload: bool, **metric_kwargs) -> None:
        val_rows = self.evaluate(
            self.val_loader,
            max_batches=max_eval_batches,
            offload_predictions_to_cpu=offload,
            **metric_kwargs,
        )
        save_best_checkpoint(
            self.best_path,
            self.model,
            self.ema,
            len(self.history),
            get_average_metric(val_rows, "mae"),
            eval_uses_ema=self.ema_for_eval is not None,
            global_step=self.global_step,
        )

    def test(
        self,
        *,
        max_eval_batches: Optional[int],
        mae_mask_value: MaskValue,
        rmse_mask_value: MaskValue,
        mape_mask_value: MaskValue,
        mape_eps: float,
    ) -> tuple[list[Dict], Dict]:
        best_load_info = load_best_checkpoint_for_test(self.model, self.best_path, self.device)
        test_rows = self.evaluate(
            self.test_loader,
            max_batches=max_eval_batches,
            mae_mask_value=mae_mask_value,
            rmse_mask_value=rmse_mask_value,
            mape_mask_value=mape_mask_value,
            mape_eps=mape_eps,
            offload_predictions_to_cpu=bool(self.eval_config.get("offload_predictions_to_cpu", False)),
        )
        write_rows_csv(self.test_metrics_path, test_rows)
        return test_rows, best_load_info
