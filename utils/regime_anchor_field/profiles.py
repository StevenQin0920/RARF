from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def temporal_ids(history: np.ndarray, tod_size: int, dow_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tod = np.floor(history[:, -1, 0, 1] * tod_size).astype(np.int64)
    tod = np.clip(tod, 0, tod_size - 1)
    dow = history[:, -1, 0, 2].astype(np.int64)
    dow = np.clip(dow, 0, dow_size - 1)
    week = dow * tod_size + tod
    return tod, dow, week


def profile_by_slot(
    series: np.ndarray,
    slots: np.ndarray,
    num_slots: int,
    *,
    fallback: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    num_nodes = series.shape[1]
    profile = np.zeros((num_slots, num_nodes), dtype=np.float32)
    counts = np.zeros(num_slots, dtype=np.int64)
    global_mean = np.nanmean(series, axis=0).astype(np.float32)
    for slot in range(num_slots):
        mask = slots == slot
        counts[slot] = int(mask.sum())
        if counts[slot] > 0:
            values = series[mask].astype(np.float32, copy=False)
            finite = np.isfinite(values)
            node_counts = finite.sum(axis=0)
            fallback_values = fallback[slot].astype(np.float32) if fallback is not None else global_mean
            sums = np.where(finite, values, 0.0).sum(axis=0)
            profile[slot] = np.divide(
                sums,
                node_counts,
                out=fallback_values.astype(np.float32, copy=True),
                where=node_counts > 0,
            ).astype(np.float32)
        elif fallback is not None:
            profile[slot] = fallback[slot].astype(np.float32)
        else:
            profile[slot] = global_mean
    return profile, counts


def expand_daily_profile_to_weekly(daily_profile: np.ndarray, dow_size: int) -> np.ndarray:
    return np.tile(daily_profile, (dow_size, 1)).astype(np.float32)


def standardize_profile(profile: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((profile.astype(np.float32) - float(mean)) / float(std)).astype(np.float32)


def scale_profile_delta(profile_delta: np.ndarray, std: float) -> np.ndarray:
    return (profile_delta.astype(np.float32) / float(std)).astype(np.float32)
