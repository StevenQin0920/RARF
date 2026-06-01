from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from utils.dataset_specs import get_dataset_spec

from .common import FEATURE_CHANNELS, _canonical_dataset_dir, _ensure_dir, _json_dump, _utc_timestamp
from .graph_assets import build_graph_artifacts
from .splits import (
    _compute_split_counts,
    _export_split_npz_artifacts,
    _resolve_artifact_mode,
    compute_train_history_stats,
)
from .time_features import build_processed_series


def prepare_dataset(
    dataset_name: str,
    raw_root: Path,
    processed_root: Path,
    split_root: Path,
    history_len: int = 12,
    horizon: int = 12,
    artifact_mode: str = "split_npz",
    npz_export_chunk_size: int = 64,
    force: bool = False,
) -> Dict:
    spec = get_dataset_spec(dataset_name)
    artifact_mode = _resolve_artifact_mode(artifact_mode)
    processed_dir = _canonical_dataset_dir(processed_root, spec.canonical_name)
    split_dir = _canonical_dataset_dir(split_root, spec.canonical_name)
    rarf_asset_dir = processed_dir / "anchors"
    _ensure_dir(processed_dir)
    _ensure_dir(split_dir)
    _ensure_dir(rarf_asset_dir)

    series_path = processed_dir / "traffic.npy"
    split_npz_paths = {mode: processed_dir / f"{mode}.npz" for mode in ("train", "val", "test")}
    dataset_metadata_path = processed_dir / "dataset_metadata.json"
    split_metadata_path = split_dir / "split_metadata.json"
    graph_metadata_path = processed_dir / "graphs" / "graph_metadata.json"

    required_paths = [dataset_metadata_path, split_metadata_path, graph_metadata_path]
    if artifact_mode in {"series", "both"}:
        required_paths.append(series_path)
    if artifact_mode in {"split_npz", "both"}:
        required_paths.extend(split_npz_paths.values())

    if not force and all(path.exists() for path in required_paths):
        return {
            "dataset": spec.canonical_name,
            "status": "skipped_existing",
            "artifact_mode": artifact_mode,
            "series_path": str(series_path) if series_path.exists() else None,
            "split_npz_paths": {
                key: str(path) for key, path in split_npz_paths.items() if path.exists()
            },
            "dataset_metadata_path": str(dataset_metadata_path),
            "split_metadata_path": str(split_metadata_path),
        }

    if artifact_mode == "series":
        for path in split_npz_paths.values():
            if path.exists():
                path.unlink()
    elif artifact_mode == "split_npz":
        if series_path.exists():
            series_path.unlink()

    series, series_metadata = build_processed_series(spec, raw_root)
    if artifact_mode in {"series", "both"}:
        np.save(series_path, series.astype(np.float32, copy=False))

    num_steps = int(series.shape[0])
    x_offsets = np.arange(0, history_len, dtype=np.int64)
    y_offsets = np.arange(history_len, history_len + horizon, dtype=np.int64)
    num_windows = int(num_steps - history_len - horizon + 1)
    if num_windows <= 0:
        raise ValueError(
            f"{spec.canonical_name} has insufficient length {num_steps} for history={history_len}, horizon={horizon}."
        )
    split_counts = _compute_split_counts(num_windows, spec.split_ratios)
    split_ranges = {
        "train": {"start": 0, "end": split_counts["train"]},
        "val": {
            "start": split_counts["train"],
            "end": split_counts["train"] + split_counts["val"],
        },
        "test": {
            "start": split_counts["train"] + split_counts["val"],
            "end": num_windows,
        },
    }
    scaler_stats = compute_train_history_stats(series, split_counts["train"], history_len)
    graph_metadata = build_graph_artifacts(spec, raw_root, processed_root)

    temporal_feature_metadata = dict(series_metadata["temporal_feature_metadata"])
    temporal_feature_metadata.update(
        {
            "feature_channels": list(FEATURE_CHANNELS),
            "feature_channel_indices": {name: idx for idx, name in enumerate(FEATURE_CHANNELS)},
            "feature_dtypes": {
                "traffic": "float32",
                "time_in_day": "float32",
                "day_of_week": "float32 integer-coded 0..6",
                "is_weekend": "float32 binary",
                "is_holiday": "float32 binary",
            },
        }
    )

    split_npz_inventory = None
    if artifact_mode in {"split_npz", "both"}:
        split_npz_inventory = _export_split_npz_artifacts(
            series=series,
            split_ranges=split_ranges,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            processed_dir=processed_dir,
            npz_export_chunk_size=npz_export_chunk_size,
        )

    split_metadata = {
        "dataset": spec.canonical_name,
        "created_at_utc": _utc_timestamp(),
        "processing_mode": (
            "fixed_offset_series_plus_split_npz"
            if artifact_mode == "both"
            else ("fixed_offset_split_npz" if artifact_mode == "split_npz" else "fixed_offset_sliding_windows_on_the_fly")
        ),
        "series_path": str(series_path) if artifact_mode in {"series", "both"} else None,
        "shape_raw_series": list(series.shape),
        "num_examples": num_windows,
        "history_length": history_len,
        "horizon": horizon,
        "x_offsets": x_offsets.tolist(),
        "y_offsets": y_offsets.tolist(),
        "split_counts": split_counts,
        "split_ratios_requested": {
            "train": spec.split_ratios[0],
            "val": spec.split_ratios[1],
            "test": spec.split_ratios[2],
        },
        "split_ranges": split_ranges,
        "feature_channels": list(FEATURE_CHANNELS),
        "feature_channel_indices": {name: idx for idx, name in enumerate(FEATURE_CHANNELS)},
        "target_feature_name": spec.target_feature_name,
        "temporal_feature_metadata": temporal_feature_metadata,
        "artifact_mode_requested": artifact_mode,
        "available_artifacts": {
            "series": artifact_mode in {"series", "both"},
            "split_npz": artifact_mode in {"split_npz", "both"},
        },
        "split_npz_paths": None if split_npz_inventory is None else split_npz_inventory["split_npz_paths"],
        "train_scaler": {
            "type": "standard",
            "applied_on": "channel_0_history_windows_only",
            **scaler_stats,
        },
    }
    _json_dump(split_metadata_path, split_metadata)

    dataset_metadata = {
        "dataset": spec.canonical_name,
        "created_at_utc": _utc_timestamp(),
        "raw_dir": str(_canonical_dataset_dir(raw_root, spec.canonical_name)),
        "processed_dir": str(processed_dir),
        "split_dir": str(split_dir),
        "graph_dir": str(processed_dir / "graphs"),
        "rarf_asset_dir": str(rarf_asset_dir),
        "seq_len": history_len,
        "horizon": horizon,
        "num_nodes": spec.num_nodes,
        "num_timesteps": num_steps,
        "processing_mode": split_metadata["processing_mode"],
        "reference_style": "fixed-offset benchmark windows with RARF temporal covariates",
        "feature_channels": list(FEATURE_CHANNELS),
        "feature_channel_indices": {name: idx for idx, name in enumerate(FEATURE_CHANNELS)},
        "target_feature_name": spec.target_feature_name,
        "target_channel_index": 0,
        "raw_feature_names": list(spec.raw_feature_names),
        "traffic_primary_channel_index": spec.primary_channel_index,
        "train_ratio": spec.split_ratios[0],
        "val_ratio": spec.split_ratios[1],
        "test_ratio": spec.split_ratios[2],
        "timezone_used": spec.timezone,
        "temporal_feature_metadata": temporal_feature_metadata,
        "split_metadata": split_metadata,
        "graph_metadata": graph_metadata,
        "train_scaler": split_metadata["train_scaler"],
        "raw": series_metadata["raw"],
        "artifact_mode_requested": artifact_mode,
        "artifact_inventory": {
            "series": {
                "saved": artifact_mode in {"series", "both"},
                "path": str(series_path) if artifact_mode in {"series", "both"} else None,
            },
            "split_npz": {
                "saved": artifact_mode in {"split_npz", "both"},
                "paths": None if split_npz_inventory is None else split_npz_inventory["split_npz_paths"],
                "shapes": None if split_npz_inventory is None else split_npz_inventory["split_npz_shapes"],
                "estimated_uncompressed_bytes": None
                if split_npz_inventory is None
                else split_npz_inventory["split_npz_estimated_uncompressed_bytes"],
                "chunk_size": None if split_npz_inventory is None else split_npz_inventory["npz_export_chunk_size"],
            },
        },
        "notes": [
            "Window generation uses fixed offsets: 12-step history to 12-step horizon.",
            "Channel 0 keeps the dataset primary traffic value; RARF temporal covariates occupy channels 1..4.",
            "Runtime normalization is unified to a train-history standard scaler on channel 0 for all datasets.",
        ],
    }
    if spec.raw_kind == "npz":
        dataset_metadata["notes"].append(
            "Raw NPZ datasets do not include explicit timestamps; weekend and holiday features are derived from a synthetic local calendar anchored at an assumed benchmark start date."
        )
    if artifact_mode in {"split_npz", "both"}:
        dataset_metadata["notes"].append(
            "Split artifacts were exported as train/val/test.npz for reproducible RARF training."
        )
    _json_dump(dataset_metadata_path, dataset_metadata)

    return {
        "dataset": spec.canonical_name,
        "status": "prepared",
        "artifact_mode": artifact_mode,
        "series_path": str(series_path) if artifact_mode in {"series", "both"} else None,
        "split_npz_paths": None if split_npz_inventory is None else split_npz_inventory["split_npz_paths"],
        "dataset_metadata_path": str(dataset_metadata_path),
        "split_metadata_path": str(split_metadata_path),
        "graph_metadata_path": str(processed_dir / "graphs" / "graph_metadata.json"),
    }


def prepare_datasets(
    dataset_names: Iterable[str],
    raw_root: Path,
    processed_root: Path,
    split_root: Path,
    history_len: int = 12,
    horizon: int = 12,
    artifact_mode: str = "split_npz",
    npz_export_chunk_size: int = 64,
    force: bool = False,
) -> List[Dict]:
    results = []
    for dataset_name in dataset_names:
        results.append(
            prepare_dataset(
                dataset_name=dataset_name,
                raw_root=raw_root,
                processed_root=processed_root,
                split_root=split_root,
                history_len=history_len,
                horizon=horizon,
                artifact_mode=artifact_mode,
                npz_export_chunk_size=npz_export_chunk_size,
                force=force,
            )
        )
    return results
