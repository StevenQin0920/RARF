from __future__ import annotations

import argparse

from utils.config import load_config, resolve_runtime_config, resolve_target_policy
from utils.experiment import (
    build_experiment_paths,
    build_rarf_model,
    build_trainer,
    ensure_experiment_paths,
    load_data_resources,
    resolve_device,
    resolve_run_id,
)
from utils.reporting import print_test_metrics_by_horizon
from utils.runtime import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate RARF.")
    parser.add_argument("--config", default="configs/PEMS08.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_config = load_config(args.config)
    config = resolve_runtime_config(source_config)
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
    paths = build_experiment_paths(config, run_id)
    ensure_experiment_paths(paths)

    resources = load_data_resources(config)
    model = build_rarf_model(config, target_policy, resources, device)
    trainer = build_trainer(
        config=config,
        model=model,
        resources=resources,
        device=device,
        paths=paths,
    )

    history = trainer.fit(
        epochs=epochs,
        max_train_batches=None,
        max_eval_batches=None,
        train_mask_value=target_policy["train_mask_value"],
        mae_mask_value=target_policy["mae_mask_value"],
        rmse_mask_value=target_policy["rmse_mask_value"],
        mape_mask_value=target_policy["mape_mask_value"],
        mape_eps=float(target_policy["mape_eps"]),
    )
    test_rows, best_load_info = trainer.test(
        max_eval_batches=None,
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

# python main.py --config configs/PEMS03.json --device cuda --run-id pems03_rarf_seed1
# python main.py --config configs/PEMS04.json --device cuda --run-id pems04_rarf_seed1
# python main.py --config configs/PEMS07.json --device cuda --run-id pems07_rarf_seed1
# python main.py --config configs/PEMS08.json --device cuda --run-id pems08_rarf_seed1

# python scripts/train_all.py --profile nofft --device cuda
# python scripts/train_all.py --profile fft001 --device cuda
# python scripts/train_all.py --profile both --device cuda