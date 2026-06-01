from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation.rarf_ablation import ABLATION_VARIANTS, RARFAblation, normalize_ablation_variant  # noqa: E402
from engine import RARFTrainer  # noqa: E402
from utils.config import build_model_kwargs, load_config, resolve_runtime_config, resolve_target_policy  # noqa: E402
from utils.experiment import load_data_resources, resolve_device, resolve_run_id  # noqa: E402
from utils.reporting import print_test_metrics_by_horizon  # noqa: E402
from utils.runtime import ensure_dir, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train isolated RARF component ablation variants.")
    parser.add_argument("--config", required=True, help="RARF slim config path.")
    parser.add_argument("--variant", required=True, choices=sorted(ABLATION_VARIANTS), help="Ablation variant.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override.")
    parser.add_argument("--run-id", default=None, help="Run id for the output directory.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches per epoch.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Limit val/test batches.")
    return parser.parse_args()


def build_ablation_run_dir(config: Dict[str, Any], variant: str, run_id: str) -> Path:
    dataset = str(config["data"]["dataset"])
    seed = int(config["train"].get("seed", 1))
    output_root = Path(config["output"].get("root", "output"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return output_root / "runs" / dataset / "RARF_ABLATION" / variant / f"seed_{seed}" / run_id


def apply_variant_runtime_overrides(config: Dict[str, Any], variant: str) -> None:
    if variant == "no-fft-loss":
        config["train"]["fft_loss_weight"] = 0.0


def build_ablation_model(
    config: Dict[str, Any],
    target_policy: Dict[str, Any],
    resources,
    device: torch.device,
    *,
    variant: str,
) -> RARFAblation:
    data_cfg = config["data"]
    missing_zero_value = float(data_cfg.get("missing_zero_value", 0.0))
    if bool(config["model"].get("use_input_missing_mask", False)) and target_policy["input_missing_mask_policy"] == "zero_as_missing":
        missing_zero_value = float(resources.scaler.transform(np.asarray([0.0], dtype=np.float32))[0])
    model_kwargs = build_model_kwargs(
        config,
        prior_adj=resources.prior_adj,
        physical_adj=resources.physical_adj,
        target_zero_is_valid=bool(target_policy["target_zero_is_valid"]),
        input_missing_mask_policy=str(target_policy["input_missing_mask_policy"]),
        missing_zero_value=missing_zero_value,
    )
    return RARFAblation(ablation_variant=variant, **model_kwargs).to(device)


def main() -> int:
    args = parse_args()
    variant = normalize_ablation_variant(args.variant)
    source_config = load_config(args.config)
    config = resolve_runtime_config(source_config)
    apply_variant_runtime_overrides(config, variant)
    target_policy = resolve_target_policy(config)

    train_cfg = config["train"]
    seed = int(train_cfg.get("seed", 1))
    set_seed(
        seed,
        deterministic=bool(train_cfg.get("deterministic", True)),
        deterministic_algorithms=bool(train_cfg.get("deterministic_algorithms", False)),
        deterministic_warn_only=bool(train_cfg.get("deterministic_warn_only", True)),
        cudnn_benchmark=bool(train_cfg.get("cudnn_benchmark", False)),
        torch_num_threads=train_cfg.get("torch_num_threads"),
    )
    device = resolve_device(args.device or train_cfg.get("device", "auto"))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 1))
    run_id = resolve_run_id(args.run_id)
    run_dir = build_ablation_run_dir(config, variant, run_id)
    ensure_dir(run_dir)

    resources = load_data_resources(config)
    model = build_ablation_model(config, target_policy, resources, device, variant=variant)
    trainer = RARFTrainer(
        model=model,
        train_loader=resources.train_loader,
        val_loader=resources.val_loader,
        test_loader=resources.test_loader,
        scaler=resources.scaler,
        device=device,
        train_config=config["train"],
        eval_config=config["eval"],
        run_dir=run_dir,
    )

    print(f"variant: {variant}")
    print(f"run_dir: {run_dir}")
    print(f"device: {device}")
    print(f"fft_loss_weight: {float(config['train'].get('fft_loss_weight', 0.0))}")
    trainer.fit(
        epochs=epochs,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        train_mask_value=target_policy["train_mask_value"],
        mae_mask_value=target_policy["mae_mask_value"],
        rmse_mask_value=target_policy["rmse_mask_value"],
        mape_mask_value=target_policy["mape_mask_value"],
        mape_eps=float(target_policy["mape_eps"]),
    )
    test_rows, best_load_info = trainer.test(
        max_eval_batches=args.max_eval_batches,
        mae_mask_value=target_policy["mae_mask_value"],
        rmse_mask_value=target_policy["rmse_mask_value"],
        mape_mask_value=target_policy["mape_mask_value"],
        mape_eps=float(target_policy["mape_eps"]),
    )
    print(f"saved checkpoint: {trainer.best_path}")
    print_test_metrics_by_horizon(test_rows, best_load_info.get("test_weight_source", "unknown_weights"))
    print(f"saved test metrics by horizon: {trainer.test_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
