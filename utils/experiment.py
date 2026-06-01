from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from dataloader import get_dataloader
from engine import RARFTrainer
from models import RARF

from .config import build_model_kwargs, resolve_path
from .io import load_adj_matrix
from .runtime import ensure_dir


@dataclass(frozen=True)
class ExperimentPaths:
    run_dir: Path


@dataclass(frozen=True)
class DataResources:
    train_loader: Any
    val_loader: Any
    test_loader: Any
    scaler: Any
    loaded_data_artifact_mode: str
    prior_adj: Any
    physical_adj: Any


def resolve_device(requested: str) -> torch.device:
    device_name = requested
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    return torch.device(device_name)


def resolve_run_id(requested_run_id: str | None) -> str:
    return requested_run_id or time.strftime("%Y%m%d_%H%M%S")


def build_experiment_paths(config: Dict[str, Any], run_id: str) -> ExperimentPaths:
    data_cfg = config["data"]
    output_cfg = config["output"]
    train_cfg = config["train"]
    dataset = str(data_cfg["dataset"])
    seed = int(train_cfg.get("seed", 1))
    output_root = Path(output_cfg.get("root", "output"))
    run_dir = output_root / "runs" / dataset / "RARF" / f"seed_{seed}" / run_id
    return ExperimentPaths(
        run_dir=run_dir,
    )


def ensure_experiment_paths(paths: ExperimentPaths) -> None:
    ensure_dir(paths.run_dir)


def load_data_resources(config: Dict[str, Any]) -> DataResources:
    data_cfg = config["data"]
    train_cfg = config["train"]
    dataset = str(data_cfg["dataset"])
    train_loader, val_loader, test_loader, scaler, _ = get_dataloader(
        dataset=dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        valid_batch_size=int(train_cfg.get("valid_batch_size", train_cfg.get("batch_size", 32))),
        test_batch_size=int(train_cfg.get("test_batch_size", train_cfg.get("batch_size", 32))),
        dataset_root=str(resolve_path(data_cfg.get("dataset_root", "datasets"))),
        expected_num_nodes=int(data_cfg["num_nodes"]),
        expected_num_features=int(data_cfg.get("input_feature_dim", int(data_cfg.get("num_feat", 1)) + 1)),
        data_artifact_mode=str(data_cfg.get("data_artifact_mode", "split_npz")),
        seed=int(train_cfg.get("seed", 1)),
    )
    loaded_data_artifact_mode = getattr(
        train_loader,
        "data_artifact_mode",
        data_cfg.get("data_artifact_mode", "split_npz"),
    )
    return DataResources(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        loaded_data_artifact_mode=loaded_data_artifact_mode,
        prior_adj=load_adj_matrix(data_cfg["prior_graph_path"]),
        physical_adj=load_adj_matrix(data_cfg["physical_graph_path"]),
    )


def build_rarf_model(
    config: Dict[str, Any],
    target_policy: Dict[str, Any],
    resources: DataResources,
    device: torch.device,
) -> RARF:
    data_cfg = config["data"]
    missing_zero_value = float(data_cfg.get("missing_zero_value", 0.0))
    if bool(config["model"].get("use_input_missing_mask", False)) and target_policy["input_missing_mask_policy"] == "zero_as_missing":
        missing_zero_value = float(resources.scaler.transform(np.asarray([0.0], dtype=np.float32))[0])
    return RARF(
        **build_model_kwargs(
            config,
            prior_adj=resources.prior_adj,
            physical_adj=resources.physical_adj,
            target_zero_is_valid=bool(target_policy["target_zero_is_valid"]),
            input_missing_mask_policy=str(target_policy["input_missing_mask_policy"]),
            missing_zero_value=missing_zero_value,
        )
    ).to(device)


def build_trainer(
    *,
    config: Dict[str, Any],
    model: RARF,
    resources: DataResources,
    device: torch.device,
    paths: ExperimentPaths,
) -> RARFTrainer:
    return RARFTrainer(
        model=model,
        train_loader=resources.train_loader,
        val_loader=resources.val_loader,
        test_loader=resources.test_loader,
        scaler=resources.scaler,
        device=device,
        train_config=config["train"],
        eval_config=config["eval"],
        run_dir=paths.run_dir,
    )
