from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .common import ARTIFACT_MODES, _ensure_dir


def _compute_split_counts(num_windows: int, split_ratios: Tuple[float, float, float]) -> Dict[str, int]:
    train_ratio, val_ratio, test_ratio = split_ratios
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {split_ratios}")
    num_test = int(round(num_windows * test_ratio))
    num_train = int(round(num_windows * train_ratio))
    num_val = int(num_windows - num_train - num_test)
    if min(num_train, num_val, num_test) <= 0:
        raise ValueError(
            f"Invalid split counts for {num_windows} windows with ratios {split_ratios}: "
            f"train={num_train}, val={num_val}, test={num_test}"
        )
    return {"train": num_train, "val": num_val, "test": num_test}


def _build_history_window_counts(total_steps: int, num_train_windows: int, history_len: int) -> np.ndarray:
    counts = np.zeros(total_steps, dtype=np.float64)
    for idx in range(total_steps):
        start_lower = max(0, idx - history_len + 1)
        start_upper = min(idx, num_train_windows - 1)
        if start_upper >= start_lower:
            counts[idx] = start_upper - start_lower + 1
    return counts


def compute_train_history_stats(series: np.ndarray, num_train_windows: int, history_len: int) -> Dict[str, float]:
    traffic = series[..., 0].astype(np.float64, copy=False)
    counts = _build_history_window_counts(traffic.shape[0], num_train_windows, history_len)[:, None]
    finite_mask = np.isfinite(traffic)
    weighted_counts = counts * finite_mask
    denom = float(weighted_counts.sum())
    if denom <= 0:
        raise ValueError("Cannot fit scaler statistics: no finite training history observations were found.")
    weighted_values = np.where(finite_mask, traffic, 0.0) * counts
    mean = float(weighted_values.sum() / denom)
    mean_sq = float((np.where(finite_mask, traffic ** 2, 0.0) * counts).sum() / denom)
    var = max(mean_sq - mean * mean, 1e-12)
    std = math.sqrt(var)
    return {
        "traffic_history_window_mean": mean,
        "traffic_history_window_std": std,
        "traffic_history_window_min": float(np.nanmin(traffic)),
        "traffic_history_window_max": float(np.nanmax(traffic)),
        "history_window_observation_count": int(round(denom)),
    }


def _resolve_artifact_mode(artifact_mode: str) -> str:
    normalized = str(artifact_mode).strip().lower()
    if normalized not in ARTIFACT_MODES:
        raise ValueError(
            f"Unsupported artifact_mode `{artifact_mode}`. Valid modes: {sorted(ARTIFACT_MODES)}"
        )
    return normalized


def _window_tensor_shape(num_windows: int, window_len: int, num_nodes: int, num_features: int) -> Tuple[int, int, int, int]:
    return (int(num_windows), int(window_len), int(num_nodes), int(num_features))


def _extract_window_chunk(
    series: np.ndarray,
    window_start: int,
    count: int,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    starts = np.arange(window_start, window_start + count, dtype=np.int64)
    x_positions = starts[:, None] + x_offsets[None, :]
    y_positions = starts[:, None] + y_offsets[None, :]
    x = np.asarray(series[x_positions, ...], dtype=np.float32)
    y = np.asarray(series[y_positions, ...], dtype=np.float32)
    return x, y


def _export_split_npz_artifacts(
    series: np.ndarray,
    split_ranges: Dict[str, Dict[str, int]],
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    processed_dir: Path,
    npz_export_chunk_size: int,
) -> Dict:
    from numpy.lib.format import open_memmap

    tmp_dir = processed_dir / "_tmp_split_npz_export"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    _ensure_dir(tmp_dir)

    num_nodes = int(series.shape[1])
    num_features = int(series.shape[2])
    artifact_paths = {}
    split_shapes = {}
    split_estimated_bytes = {}
    try:
        for split_name, split_range in split_ranges.items():
            window_start = int(split_range["start"])
            window_end = int(split_range["end"])
            num_windows = max(window_end - window_start, 0)
            x_shape = _window_tensor_shape(num_windows, len(x_offsets), num_nodes, num_features)
            y_shape = _window_tensor_shape(num_windows, len(y_offsets), num_nodes, num_features)
            split_shapes[split_name] = {"x": list(x_shape), "y": list(y_shape)}
            split_estimated_bytes[split_name] = {
                "x": int(np.prod(x_shape, dtype=np.int64) * np.dtype(np.float32).itemsize),
                "y": int(np.prod(y_shape, dtype=np.int64) * np.dtype(np.float32).itemsize),
            }

            x_tmp_path = tmp_dir / f"{split_name}_x.npy"
            y_tmp_path = tmp_dir / f"{split_name}_y.npy"
            x_tmp = open_memmap(x_tmp_path, mode="w+", dtype=np.float32, shape=x_shape)
            y_tmp = open_memmap(y_tmp_path, mode="w+", dtype=np.float32, shape=y_shape)
            try:
                for offset in range(0, num_windows, max(int(npz_export_chunk_size), 1)):
                    chunk_end = min(offset + max(int(npz_export_chunk_size), 1), num_windows)
                    x_chunk, y_chunk = _extract_window_chunk(
                        series=series,
                        window_start=window_start + offset,
                        count=chunk_end - offset,
                        x_offsets=x_offsets,
                        y_offsets=y_offsets,
                    )
                    x_tmp[offset:chunk_end, ...] = x_chunk
                    y_tmp[offset:chunk_end, ...] = y_chunk
            finally:
                del x_tmp
                del y_tmp

            split_path = processed_dir / f"{split_name}.npz"
            x_memmap = np.load(x_tmp_path, mmap_mode="r")
            y_memmap = np.load(y_tmp_path, mmap_mode="r")
            try:
                np.savez_compressed(split_path, x=x_memmap, y=y_memmap)
            finally:
                del x_memmap
                del y_memmap
            artifact_paths[split_name] = str(split_path)
        return {
            "split_npz_paths": artifact_paths,
            "split_npz_shapes": split_shapes,
            "split_npz_estimated_uncompressed_bytes": split_estimated_bytes,
            "npz_export_chunk_size": int(npz_export_chunk_size),
        }
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
