from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from utils.dataset_specs import canonicalize_dataset_name


VALID_DATA_ARTIFACT_MODES = {"auto", "series", "split_npz"}
PREPARE_DATA_COMMAND = "python -m utils.prepare_data --datasets {dataset} --artifact-mode split_npz"


def prepare_data_command(dataset: str) -> str:
    return PREPARE_DATA_COMMAND.format(dataset=canonicalize_dataset_name(dataset))


def resolve_dataset_dir(dataset: str, dataset_root: str = "dataset", data_artifact_mode: str = "auto") -> tuple[Path, str]:
    canonical = canonicalize_dataset_name(dataset)
    requested_mode = str(data_artifact_mode).strip().lower()
    if requested_mode not in VALID_DATA_ARTIFACT_MODES:
        raise ValueError(
            f"Unsupported data_artifact_mode `{data_artifact_mode}`. "
            f"Valid modes: {sorted(VALID_DATA_ARTIFACT_MODES)}"
        )
    root = Path(dataset_root)
    candidates = []
    if root.name == canonical and root.exists():
        candidates.append(root)
    else:
        candidates.append(root / canonical)
    project_processed = Path(__file__).resolve().parents[1] / "datasets" / canonical
    candidates.append(project_processed)

    for candidate in candidates:
        has_split_npz = (candidate / "train.npz").exists()
        has_series = (candidate / "traffic.npy").exists() and (candidate / "dataset_metadata.json").exists()
        if requested_mode == "series" and has_series:
            return candidate, "series"
        if requested_mode == "split_npz" and has_split_npz:
            return candidate, "split_npz"
        if requested_mode == "auto":
            if has_split_npz:
                return candidate, "split_npz"
            if has_series:
                return candidate, "series"
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Dataset artifacts for `{canonical}` with mode `{requested_mode}` were not found. Searched: {searched}. "
        f"Generate them with: {prepare_data_command(canonical)}"
    )


def load_split_arrays(dataset_dir: Path, mode: str) -> Dict[str, np.ndarray]:
    split_path = dataset_dir / f"{mode}.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_path}. "
            f"Generate datasets with: {prepare_data_command(dataset_dir.name)}"
        )
    with np.load(split_path) as split:
        return {"x": split["x"], "y": split["y"]}


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def validate_feature_shapes(
    feature_count: int,
    expected_num_nodes: Optional[int],
    expected_num_features: Optional[int],
    num_nodes: int,
) -> None:
    if expected_num_nodes is not None and num_nodes != expected_num_nodes:
        raise ValueError(f"Node count {num_nodes} != expected {expected_num_nodes}")
    if expected_num_features is not None and feature_count < expected_num_features:
        raise ValueError(f"Feature count {feature_count} < expected {expected_num_features}")
