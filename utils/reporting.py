from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable

from .runtime import ensure_dir

def format_epoch_log(row: Dict) -> str:
    parts = [
        f"epoch={row['epoch']}",
        "model=RARF",
        f"train_mae={row['train_mae']:.5f}",
        f"val_mae={row['val_mae']:.5f}",
        f"best_val={row['best_val_mae']:.5f}",
        f"lr={row['lr']:.6g}",
        f"time={row['epoch_time_sec']:.1f}s",
    ]
    for horizon in (3, 6, 12):
        key = f"val_h{horizon}_mae"
        if key in row:
            parts.append(f"val_mae@{horizon}={row[key]:.4f}")
    for key, label in (
        ("train_fft_loss", "train_fft"),
        ("fft_loss_weight", "fft_w"),
        ("skipped_batches", "skipped"),
    ):
        if key in row and row[key] is not None:
            parts.append(f"{label}={row[key]:.4f}")
    if row.get("is_best"):
        parts.append("best=*")
    if row.get("early_stop"):
        parts.append("early_stop=True")
    return " ".join(parts)

def write_history_csv(path: Path, history: Iterable[Dict]) -> None:
    history = list(history)
    if not history:
        return
    ensure_dir(path.parent)
    base_fieldnames = [
        "epoch",
        "train_mae",
        "val_mae",
        "val_rmse",
        "val_mape",
        "best_val_mae",
        "best_epoch",
        "lr",
        "is_best",
        "early_stop",
        "epoch_time_sec",
        "val_h3_mae",
        "val_h6_mae",
        "val_h12_mae",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            normalized = {}
            for key in base_fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False)
                normalized[key] = value
            writer.writerow(normalized)

def write_rows_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = ["horizon", "mae", "rmse", "mape"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False)
                normalized[key] = value
            writer.writerow(normalized)

def print_test_metrics_by_horizon(rows: Iterable[Dict], weight_source: str) -> None:
    rows = list(rows)
    horizon_rows = [row for row in rows if isinstance(row.get("horizon"), int)]
    avg_row = next((row for row in rows if row.get("horizon") == "avg"), None)
    print(f"test results by horizon using {weight_source}:")
    print("horizon  mae      rmse     mape")
    for row in horizon_rows:
        print(
            f"{int(row['horizon']):>7d} "
            f"{float(row['mae']):>8.4f} "
            f"{float(row['rmse']):>8.4f} "
            f"{float(row['mape']):>8.4f}"
        )
    if avg_row is not None:
        print(
            "avg     "
            f"{float(avg_row['mae']):>8.4f} "
            f"{float(avg_row['rmse']):>8.4f} "
            f"{float(avg_row['mape']):>8.4f}"
        )


