from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def profile_stats(profile: np.ndarray, counts: np.ndarray) -> dict:
    return {
        "shape": list(profile.shape),
        "slot_count_min": int(counts.min()),
        "slot_count_mean": float(counts.mean()),
        "slot_count_max": int(counts.max()),
        "value_min": float(np.nanmin(profile)),
        "value_mean": float(np.nanmean(profile)),
        "value_max": float(np.nanmax(profile)),
        "has_nan": bool(np.isnan(profile).any()),
    }


def build_metadata(
    *,
    dataset: str,
    train_npz: Path,
    daily_path: Path,
    weekly_path: Path,
    history: np.ndarray,
    series: np.ndarray,
    tod_slots: np.ndarray,
    dow_slots: np.ndarray,
    week_slots: np.ndarray,
    tod_size: int,
    dow_size: int,
    scaler_mean: float,
    scaler_std: float,
    scaler_source: str,
    daily_profile_raw: np.ndarray,
    weekly_mean_raw: np.ndarray,
    weekly_residual_raw: np.ndarray,
    daily_profile: np.ndarray,
    weekly_profile: np.ndarray,
    daily_counts: np.ndarray,
    weekly_counts: np.ndarray,
) -> dict:
    return {
        "dataset": dataset,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "built_train_only_daily_weekly_residual_regime_anchor_fields",
        "method": "rarf_regime_anchor_field",
        "description": (
            "Train-only target-channel daily anchors and weekly day-specific deviations "
            "saved in normalized model scale."
        ),
        "anchor_estimator": "mean_raw",
        "weekly_residual_mode": "raw",
        "anchor_definitions": {
            "daily_profile_raw": "A_daily(n,tau) = mean_train(y | node=n, TOD=tau)",
            "weekly_mean_raw": "A_daily(n,tau) + A_weekly_res(n,tau,d)",
            "weekly_residual_raw": "A_weekly_res(n,tau,d) = weekly_mean_raw(n,tau,d) - A_daily(n,tau)",
            "regime_anchor": "A0(n,tau,d) = A_daily(n,tau) + A_weekly_res(n,tau,d)",
            "paper_sentence": "The weekly regime anchor is defined as a day-specific deviation from the daily regime anchor.",
        },
        "train_npz": str(train_npz),
        "daily_init_path": str(daily_path),
        "weekly_init_path": str(weekly_path),
        "num_train_examples": int(history.shape[0]),
        "num_nodes": int(history.shape[2]),
        "tod_size": int(tod_size),
        "dow_size": int(dow_size),
        "weekly_size": int(tod_size * dow_size),
        "normalization": {
            "status": "applied_before_save",
            "daily_formula": "(daily_profile_raw - mean) / std",
            "weekly_residual_formula": "weekly_residual_raw / std",
            "reconstruction_formula": "daily_profile[tod] + weekly_profile[dow * tod_size + tod]",
            "mean": float(scaler_mean),
            "std": float(scaler_std),
            "source": scaler_source,
        },
        "daily_profile_raw": profile_stats(daily_profile_raw, daily_counts),
        "weekly_mean_raw": profile_stats(weekly_mean_raw, weekly_counts),
        "weekly_residual_raw": profile_stats(weekly_residual_raw, weekly_counts),
        "daily_profile": profile_stats(daily_profile, daily_counts),
        "weekly_profile": profile_stats(weekly_profile, weekly_counts),
        "data_tensor_hash": array_hash(history),
        "traffic_series_hash": array_hash(series),
        "tod_slots_hash": array_hash(tod_slots),
        "dow_slots_hash": array_hash(dow_slots),
        "week_slots_hash": array_hash(week_slots),
        "daily_profile_hash": array_hash(daily_profile),
        "weekly_profile_hash": array_hash(weekly_profile),
        "daily_profile_raw_hash": array_hash(daily_profile_raw),
        "weekly_mean_raw_hash": array_hash(weekly_mean_raw),
        "weekly_residual_raw_hash": array_hash(weekly_residual_raw),
        "leakage_guard": "train_split_only_no_val_or_test_statistics",
    }
