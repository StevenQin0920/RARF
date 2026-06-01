from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .artifacts import load_json, load_split_arrays, resolve_dataset_dir, validate_feature_shapes
from .datasets import ArrayTrafficDataset, WindowedSeriesDataset
from .scaler import StandardScaler


def _build_series_loaders(
    dataset_dir: Path,
    batch_size: int,
    valid_batch_size: int,
    test_batch_size: int,
    expected_num_nodes: Optional[int],
    expected_num_features: Optional[int],
    seed: Optional[int],
):
    dataset_metadata = load_json(dataset_dir / "dataset_metadata.json")
    split_metadata = dataset_metadata.get("split_metadata") or load_json(
        dataset_dir.parent.parent / "splits" / dataset_dir.name / "split_metadata.json"
    )
    train_scaler = split_metadata.get("train_scaler", {})
    scaler = StandardScaler(
        mean=float(train_scaler["traffic_history_window_mean"]),
        std=float(train_scaler["traffic_history_window_std"]),
    )
    if not np.isfinite(scaler.std) or scaler.std <= 0:
        raise ValueError(f"Invalid traffic scaler std for {dataset_dir.name}: {scaler.std}")

    series = np.load(dataset_dir / "traffic.npy", mmap_mode="r")
    validate_feature_shapes(
        feature_count=int(series.shape[-1]),
        expected_num_nodes=expected_num_nodes,
        expected_num_features=expected_num_features,
        num_nodes=int(series.shape[1]),
    )
    x_offsets = np.asarray(split_metadata["x_offsets"], dtype=np.int64)
    y_offsets = np.asarray(split_metadata["y_offsets"], dtype=np.int64)
    split_ranges = split_metadata["split_ranges"]

    train_loader = WindowedSeriesDataset(
        series=series,
        window_start=int(split_ranges["train"]["start"]),
        window_end=int(split_ranges["train"]["end"]),
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        batch_size=batch_size,
        scaler=scaler,
        shuffle=True,
        seed=seed,
    )
    val_loader = WindowedSeriesDataset(
        series=series,
        window_start=int(split_ranges["val"]["start"]),
        window_end=int(split_ranges["val"]["end"]),
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        batch_size=valid_batch_size,
        scaler=scaler,
        shuffle=False,
    )
    test_loader = WindowedSeriesDataset(
        series=series,
        window_start=int(split_ranges["test"]["start"]),
        window_end=int(split_ranges["test"]["end"]),
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        batch_size=test_batch_size,
        scaler=scaler,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, scaler, test_loader.original_size


def _build_prewindowed_loaders(
    dataset_dir: Path,
    batch_size: int,
    valid_batch_size: int,
    test_batch_size: int,
    expected_num_nodes: Optional[int],
    expected_num_features: Optional[int],
    seed: Optional[int],
):
    data_dict = {}
    for mode in ["train", "val", "test"]:
        split = load_split_arrays(dataset_dir, mode)
        data_dict[f"x_{mode}"] = split["x"]
        data_dict[f"y_{mode}"] = split["y"]

    for mode in ["train", "val", "test"]:
        x = data_dict[f"x_{mode}"]
        y = data_dict[f"y_{mode}"]
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError(f"{mode} split must be rank-4 x/y arrays, got x={x.shape}, y={y.shape}")
        validate_feature_shapes(
            feature_count=int(x.shape[-1]),
            expected_num_nodes=expected_num_nodes,
            expected_num_features=expected_num_features,
            num_nodes=int(x.shape[2]),
        )
        if expected_num_nodes is not None and y.shape[2] != expected_num_nodes:
            raise ValueError(f"{mode} y node count {y.shape[2]} != expected {expected_num_nodes}")
        if expected_num_features is not None and y.shape[-1] < expected_num_features:
            raise ValueError(f"{mode} y feature count {y.shape[-1]} < expected {expected_num_features}")

    train_traffic = data_dict["x_train"][..., 0]
    scaler_metadata = None
    dataset_metadata_path = dataset_dir / "dataset_metadata.json"
    if dataset_metadata_path.exists():
        dataset_metadata = load_json(dataset_metadata_path)
        scaler_metadata = dataset_metadata.get("train_scaler")
        if scaler_metadata is None:
            split_metadata = dataset_metadata.get("split_metadata")
            if isinstance(split_metadata, dict):
                scaler_metadata = split_metadata.get("train_scaler")
    if scaler_metadata is not None:
        scaler = StandardScaler(
            mean=float(scaler_metadata["traffic_history_window_mean"]),
            std=float(scaler_metadata["traffic_history_window_std"]),
        )
    else:
        scaler = StandardScaler(mean=np.nanmean(train_traffic), std=np.nanstd(train_traffic))
    if not np.isfinite(scaler.std) or scaler.std <= 0:
        raise ValueError(f"Invalid traffic scaler std for {dataset_dir.name}: {scaler.std}")
    for mode in ["train", "val", "test"]:
        data_dict[f"x_{mode}"] = np.asarray(data_dict[f"x_{mode}"], dtype=np.float32)
        data_dict[f"y_{mode}"] = np.asarray(data_dict[f"y_{mode}"], dtype=np.float32)
        data_dict[f"x_{mode}"][..., 0] = scaler.transform(data_dict[f"x_{mode}"][..., 0])
        data_dict[f"y_{mode}"][..., 0] = scaler.transform(data_dict[f"y_{mode}"][..., 0])

    train_loader = ArrayTrafficDataset(data_dict["x_train"], data_dict["y_train"], batch_size, shuffle=True, seed=seed)
    val_loader = ArrayTrafficDataset(data_dict["x_val"], data_dict["y_val"], valid_batch_size, shuffle=False)
    test_loader = ArrayTrafficDataset(data_dict["x_test"], data_dict["y_test"], test_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, scaler, test_loader.original_size


def get_dataloader(
    dataset,
    batch_size,
    valid_batch_size,
    test_batch_size,
    dataset_root="datasets",
    expected_num_nodes: Optional[int] = None,
    expected_num_features: Optional[int] = None,
    data_artifact_mode: str = "split_npz",
    seed: Optional[int] = None,
):
    dataset_dir, resolved_artifact_mode = resolve_dataset_dir(dataset, dataset_root, data_artifact_mode)
    if resolved_artifact_mode == "series":
        result = _build_series_loaders(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            test_batch_size=test_batch_size,
            expected_num_nodes=expected_num_nodes,
            expected_num_features=expected_num_features,
            seed=seed,
        )
    else:
        result = _build_prewindowed_loaders(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            test_batch_size=test_batch_size,
            expected_num_nodes=expected_num_nodes,
            expected_num_features=expected_num_features,
            seed=seed,
        )
    train_loader, val_loader, test_loader, scaler, test_size = result
    for loader in (train_loader, val_loader, test_loader):
        loader.data_artifact_mode = resolved_artifact_mode
        loader.dataset_dir = str(dataset_dir)
    return train_loader, val_loader, test_loader, scaler, test_size
