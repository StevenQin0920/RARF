from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

from .metadata import build_metadata
from .profiles import (
    expand_daily_profile_to_weekly,
    profile_by_slot,
    scale_profile_delta,
    standardize_profile,
    temporal_ids,
)


def load_train_scaler(dataset_root: Path, series: np.ndarray) -> Tuple[float, float, str]:
    metadata_paths = (
        dataset_root / "dataset_metadata.json",
        dataset_root / "split_metadata.json",
    )
    for metadata_path in metadata_paths:
        if not metadata_path.exists():
            continue
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        scaler = metadata.get("train_scaler")
        if scaler is None and isinstance(metadata.get("split_metadata"), dict):
            scaler = metadata["split_metadata"].get("train_scaler")
        if scaler is not None:
            mean = float(scaler["traffic_history_window_mean"])
            std = float(scaler["traffic_history_window_std"])
            if np.isfinite(std) and std > 0:
                return mean, std, str(metadata_path)
    mean = float(np.nanmean(series))
    std = float(np.nanstd(series))
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid fallback regime anchor scaler std: {std}")
    return mean, std, "computed_from_train_series_channel_0"


def build_regime_anchor_field(
    *,
    dataset: str,
    train_npz: Path,
    anchor_asset_dir: Path,
    tod_size: int = 288,
    dow_size: int = 7,
) -> dict:
    anchor_asset_dir.mkdir(parents=True, exist_ok=True)
    train = np.load(train_npz)
    history = train["x"]
    series = history[:, -1, :, 0].astype(np.float32)
    scaler_mean, scaler_std, scaler_source = load_train_scaler(train_npz.parent, series)
    tod_slots, dow_slots, week_slots = temporal_ids(history, tod_size, dow_size)

    daily_profile_raw, daily_counts = profile_by_slot(series, tod_slots, tod_size)
    weekly_fallback = expand_daily_profile_to_weekly(daily_profile_raw, dow_size)
    weekly_mean_raw, weekly_counts = profile_by_slot(
        series,
        week_slots,
        tod_size * dow_size,
        fallback=weekly_fallback,
    )
    weekly_residual_raw = (weekly_mean_raw - weekly_fallback).astype(np.float32)

    daily_profile = standardize_profile(daily_profile_raw, scaler_mean, scaler_std)
    weekly_profile = scale_profile_delta(weekly_residual_raw, scaler_std)

    daily_path = anchor_asset_dir / "regime_anchor_field_daily.npy"
    weekly_path = anchor_asset_dir / "regime_anchor_field_weekly.npy"
    np.save(daily_path, daily_profile.astype(np.float32))
    np.save(weekly_path, weekly_profile.astype(np.float32))

    metadata = build_metadata(
        dataset=dataset,
        train_npz=train_npz,
        daily_path=daily_path,
        weekly_path=weekly_path,
        history=history,
        series=series,
        tod_slots=tod_slots,
        dow_slots=dow_slots,
        week_slots=week_slots,
        tod_size=tod_size,
        dow_size=dow_size,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        scaler_source=scaler_source,
        daily_profile_raw=daily_profile_raw,
        weekly_mean_raw=weekly_mean_raw,
        weekly_residual_raw=weekly_residual_raw,
        daily_profile=daily_profile,
        weekly_profile=weekly_profile,
        daily_counts=daily_counts,
        weekly_counts=weekly_counts,
    )
    metadata_path = anchor_asset_dir / "regime_anchor_field_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return {
        "daily_path": daily_path,
        "weekly_path": weekly_path,
        "metadata_path": metadata_path,
        "daily_profile": daily_profile,
        "weekly_profile": weekly_profile,
        "metadata": metadata,
    }
